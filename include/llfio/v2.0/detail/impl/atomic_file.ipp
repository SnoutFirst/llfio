/* Atomically replace a file's contents
(C) 2026 d3fault <d3fault@d3fault.net> (1 commit)
File Created: Oct 2026


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/

#include "../../atomic_file.hpp"

LLFIO_V2_NAMESPACE_BEGIN

LLFIO_HEADERS_ONLY_MEMFUNC_SPEC result<atomic_file> atomic_file::open(const path_handle &base, path_view_type path,
                                                                      options opts) noexcept
{
  LLFIO_EXCEPTION_TRY
  {
    if(path.empty())
    {
      return errc::invalid_argument;
    }
    path_view_type parent = path.parent_path();
    path_view_type leaf = path.filename();
    if(leaf.empty())
    {
      return errc::invalid_argument;
    }
    atomic_file ret;
    ret._fallback = opts.fallback;
    // We own only the leafname, not the whole target path. The containing directory is held as an
    // open handle, which both anchors the destination against third party path changes and lets us
    // flush the directory entry at commit time.
    ret._leafname = leaf.path();
    {
      auto dirh = parent.empty() ? directory_handle::directory(base, ".", mode::read) :
                                   directory_handle::directory(base, parent, mode::read);
      if(!dirh)
      {
        return std::move(dirh).error();
      }
      ret._dirh = std::move(dirh).value();
    }
    // Prefer a uniquely named temporary file in the same directory as the destination, which is
    // what makes the eventual replacement atomic and crash-safe.
    auto temp = file_handle::uniquely_named_file(ret._dirh, mode::write, opts._caching);
    if(temp)
    {
      // A file we created ourselves, and the only thing the destructor is ever allowed to unlink.
      ret._tempfile = std::move(temp).value();
      return ret;
    }
    if(!opts.fallback)
    {
      return std::move(temp).error();
    }
    // The caller has opted in to the non-atomic fallback: write the destination in place. Note that
    // the real destination is stored in `_destination`, never in `_tempfile`, so if any of the steps
    // below fail and this partially constructed object is destroyed, the destructor has no handle to
    // the destination and cannot delete it.
    ret._fallback = true;
    auto direct = file_handle::file(ret._dirh, ret._leafname, mode::write, creation::if_needed, opts._caching);
    if(!direct)
    {
      return std::move(direct).error();
    }
    ret._destination = std::move(direct).value();
    // Represent the new content faithfully by starting from an empty destination. This truncates
    // the destination there and then, so from this point the destination has been modified and
    // cannot be restored.
    auto trunc = ret._destination.truncate(0);
    if(!trunc)
    {
      return std::move(trunc).error();
    }
    ret._destination_modified = true;
    return ret;
  }
  LLFIO_EXCEPTION_CATCH_ALL
  {
    return error_from_exception();
  }
}

LLFIO_HEADERS_ONLY_MEMFUNC_SPEC result<atomic_file::size_type> atomic_file::write(const_buffers_type buffers,
                                                                                  deadline d) noexcept
{
  LLFIO_EXCEPTION_TRY
  {
    if(_state != state::open)
    {
      return errc::operation_not_permitted;
    }
    if(_write_failed)
    {
      // Remember the first write failure and never write anything further.
      return std::move(_write_error);
    }
    if(buffers.empty())
    {
      return size_type(0);
    }
    byte_io_handle::io_request<const_buffers_type> req(buffers, _offset);
    auto ret = _write_target().write(req, d);
    if(!ret)
    {
      _write_failed = true;
      _write_error = std::move(ret).error();
      return std::move(_write_error);
    }
    auto written = ret.bytes_transferred();
    _offset += written;
    return written;
  }
  LLFIO_EXCEPTION_CATCH_ALL
  {
    return error_from_exception();
  }
}

LLFIO_HEADERS_ONLY_MEMFUNC_SPEC result<atomic_file::size_type> atomic_file::write(const_buffer_type buffer,
                                                                                  deadline d) noexcept
{
  const_buffer_type buf[1] = {buffer};
  return write(const_buffers_type(buf, 1), d);
}

LLFIO_HEADERS_ONLY_MEMFUNC_SPEC result<void> atomic_file::commit(deadline d) noexcept
{
  LLFIO_EXCEPTION_TRY
  {
    if(_state == state::committed)
    {
      return errc::operation_not_permitted;
    }
    if(_state == state::cancelled || _state == state::failed)
    {
      return errc::operation_canceled;
    }
    if(_write_failed)
    {
      _state = state::failed;
      return std::move(_write_error);
    }
    // Exactly one durability barrier, at commit time, for all the data written so far plus the
    // file metadata needed to retrieve it. Note that a synchronous barrier cannot honour a
    // deadline (LLFIO returns `errc::not_supported` for that), so we pass an empty deadline here.
    // This is a barrier, not a destructive operation, so it is safe on the real destination.
    auto bar = _write_target().barrier(barrier_kind::wait_all, deadline());
    if(!bar)
    {
      _state = state::failed;
      return std::move(bar).error();
    }
    if(_fallback)
    {
      // We have been writing the destination directly all along; there is nothing to rename.
      _state = state::committed;
      return success();
    }
    // Atomically and silently replace whatever is at the destination. Fallback mode returned above,
    // so `_tempfile` is definitely the temporary file here.
    auto rel = _tempfile.relink(_dirh, _leafname, true, d);
    if(!rel)
    {
      _state = state::failed;
      return std::move(rel).error();
    }
    // The relocation succeeded, so the destination now names the new content. Record that *before*
    // anything else can fail, so that a caller which sees a failure from the directory flush (or a
    // deadline which expired once the rename had been issued) can still tell that the destination
    // was modified. See replacement_performed().
    _destination_modified = true;
    _state = state::committed;
    auto dirbar = _dirh.barrier();
    if(!dirbar)
    {
      return std::move(dirbar).error();
    }
    return success();
  }
  LLFIO_EXCEPTION_CATCH_ALL
  {
    return error_from_exception();
  }
}

LLFIO_HEADERS_ONLY_MEMFUNC_SPEC result<void> atomic_file::cancel() noexcept
{
  if(_state == state::cancelled)
  {
    return success();
  }
  if(_state == state::committed || _state == state::failed)
  {
    return errc::operation_not_permitted;
  }
  if(_fallback)
  {
    // We have been writing the destination directly. Those writes cannot be undone, so we refuse
    // to pretend that cancellation succeeded, and leave the state unchanged.
    return errc::not_supported;
  }
  _state = state::cancelled;
  return success();
}

LLFIO_V2_NAMESPACE_END