/* Integration test kernel for atomic_file
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

#include "../test_kernel_decl.hpp"

#include <atomic>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace llfio = LLFIO_V2_NAMESPACE;

using atomic_file = llfio::atomic_file;

static llfio::byte_io_handle::const_buffer_type buf(const std::string &s)
{
  return llfio::byte_io_handle::const_buffer_type(reinterpret_cast<const llfio::byte *>(s.data()), s.size());
}

static std::string read_file(const llfio::path_handle &base, llfio::path_view p)
{
  auto fh = llfio::file_handle::file(base, p);
  BOOST_REQUIRE(fh);
  auto size = fh.value().maximum_extent();
  BOOST_REQUIRE(size);
  std::string ret(size.value(), '\0');
  if(!ret.empty())
  {
    auto readed = llfio::read(fh.value(), 0, {{reinterpret_cast<llfio::byte *>(ret.data()), ret.size()}});
    BOOST_REQUIRE(readed);
    ret.resize(readed.value());
  }
  return ret;
}

static size_t count_random_files(const llfio::filesystem::path &p)
{
  size_t ret = 0;
  std::error_code ec;
  for(auto it = llfio::filesystem::directory_iterator(p, ec); !ec && it != llfio::filesystem::directory_iterator();
      it.increment(ec))
  {
    const std::string name = it->path().filename().string();
    if(name.size() > 7 && name.compare(name.size() - 7, 7, ".random") == 0)
    {
      ret++;
    }
  }
  return ret;
}

// Create a brand new file, and check its content is what we wrote.
static void TestCreateNewFile(const llfio::directory_handle &dirh)
{
  {
    atomic_file f = atomic_file::open(dirh, "newfile.txt").value();
    BOOST_CHECK(f.current_state() == atomic_file::state::open);
    BOOST_CHECK(!f.is_fallback());
    BOOST_CHECK(!f.replacement_performed());
    BOOST_CHECK(f.write(buf("hello world")).value() == 11);
    BOOST_CHECK(f.current_state() == atomic_file::state::open);
    // The destination must not exist before commit
    BOOST_CHECK(!llfio::file_handle::file(dirh, "newfile.txt"));
    BOOST_CHECK(f.commit());
    BOOST_CHECK(f.current_state() == atomic_file::state::committed);
    BOOST_CHECK(f.replacement_performed());
  }
  BOOST_CHECK(read_file(dirh, "newfile.txt") == "hello world");
}

// Overwrite an existing file, including with a shorter content (so stale trailing bytes must go).
static void TestOverwriteExisting(const llfio::directory_handle &dirh)
{
  {
    atomic_file f = atomic_file::open(dirh, "existing.txt").value();
    BOOST_CHECK(f.write(buf("0123456789")).value() == 10);
    BOOST_CHECK(f.commit());
  }
  BOOST_CHECK(read_file(dirh, "existing.txt") == "0123456789");
  {
    atomic_file f = atomic_file::open(dirh, "existing.txt").value();
    BOOST_CHECK(f.write(buf("abc")).value() == 3);
    BOOST_CHECK(f.commit());
  }
  BOOST_CHECK(read_file(dirh, "existing.txt") == "abc");
}

// Multiple sequential writes must concatenate, and must go through the tracked offset.
static void TestMultipleWrites(const llfio::directory_handle &dirh)
{
  {
    atomic_file f = atomic_file::open(dirh, "multi.txt").value();
    for(unsigned n = 0; n < 1000; n++)
    {
      std::string s = std::to_string(n) + ",";
      BOOST_CHECK(f.write(buf(s)).value() == s.size());
    }
    BOOST_CHECK(f.commit());
  }
  std::string expected;
  for(unsigned n = 0; n < 1000; n++)
  {
    expected += std::to_string(n) + ",";
  }
  BOOST_CHECK(read_file(dirh, "multi.txt") == expected);
}

// A zero length write is a no-op, not an error.
static void TestEmptyWrite(const llfio::directory_handle &dirh)
{
  {
    atomic_file f = atomic_file::open(dirh, "empty.txt").value();
    BOOST_CHECK(f.write(buf("")).value() == 0);
    BOOST_CHECK(f.commit());
  }
  BOOST_CHECK(read_file(dirh, "empty.txt").empty());
}

// Destroying without committing must leave the destination untouched, and must clean up the temp.
static void TestDiscardOnDestruction(const llfio::directory_handle &dirh)
{
  {
    atomic_file f = atomic_file::open(dirh, "discarded.txt").value();
    BOOST_CHECK(f.write(buf("this should never appear")).value() > 0);
    BOOST_CHECK(!f.replacement_performed());
    // No commit: destructor discards
  }
  BOOST_CHECK(!llfio::file_handle::file(dirh, "discarded.txt"));
  // And no temporary files may be left behind
  BOOST_CHECK(count_random_files(dirh.current_path().value()) == 0);
}

// Cancelling must prevent publication, and committing afterwards must fail.
static void TestCancel(const llfio::directory_handle &dirh)
{
  {
    atomic_file f = atomic_file::open(dirh, "cancelled.txt").value();
    BOOST_CHECK(f.write(buf("nope")).value() == 4);
    BOOST_CHECK(f.cancel());
    BOOST_CHECK(f.current_state() == atomic_file::state::cancelled);
    BOOST_CHECK(!f.replacement_performed());
    // Writing after cancellation is not permitted
    auto w = f.write(buf("more"));
    BOOST_REQUIRE(w.has_error());
    BOOST_CHECK(w.error() == llfio::errc::operation_not_permitted);
    // Committing after cancellation must not publish
    auto c = f.commit();
    BOOST_REQUIRE(c.has_error());
    BOOST_CHECK(c.error() == llfio::errc::operation_canceled);
    // Cancelling twice is harmless
    BOOST_CHECK(f.cancel());
  }
  BOOST_CHECK(!llfio::file_handle::file(dirh, "cancelled.txt"));
}

// Committing twice is not permitted.
static void TestDoubleCommit(const llfio::directory_handle &dirh)
{
  atomic_file f = atomic_file::open(dirh, "double.txt").value();
  BOOST_CHECK(f.write(buf("x")).value() == 1);
  BOOST_CHECK(f.commit());
  auto c = f.commit();
  BOOST_REQUIRE(c.has_error());
  BOOST_CHECK(c.error() == llfio::errc::operation_not_permitted);
  // And writing after commit is not permitted
  auto w = f.write(buf("y"));
  BOOST_REQUIRE(w.has_error());
  BOOST_CHECK(w.error() == llfio::errc::operation_not_permitted);
  BOOST_CHECK(read_file(dirh, "double.txt") == "x");
}

// A path with a directory component is resolved relative to the base handle.
static void TestPathWithDirectory(const llfio::directory_handle &dirh)
{
  std::error_code ec;
  llfio::filesystem::create_directory(dirh.current_path().value() / "subdir", ec);
  BOOST_REQUIRE(!ec);
  {
    atomic_file f = atomic_file::open(dirh, "subdir/leaf.txt").value();
    BOOST_CHECK(f.write(buf("in a subdirectory")).value() == 17);
    BOOST_CHECK(f.commit());
  }
  BOOST_CHECK(read_file(dirh, "subdir/leaf.txt") == "in a subdirectory");
}

// Opening into a directory which does not exist must fail.
static void TestOpenIntoNonexistentDirectory(const llfio::directory_handle &dirh)
{
  auto f = atomic_file::open(dirh, "no/such/dir/leaf.txt");
  BOOST_REQUIRE(f.has_error());
}

// Replacing a path which is actually a directory must fail at commit, and leave the directory alone.
static void TestDestinationIsDirectory(const llfio::directory_handle &dirh)
{
  std::error_code ec;
  llfio::filesystem::create_directory(dirh.current_path().value() / "adir", ec);
  BOOST_REQUIRE(!ec);
  {
    atomic_file f = atomic_file::open(dirh, "adir").value();
    BOOST_CHECK(f.write(buf("not allowed")).value() == 11);
    auto c = f.commit();
    BOOST_REQUIRE(c.has_error());
    BOOST_CHECK(!f.replacement_performed());
  }
  BOOST_CHECK(llfio::filesystem::is_directory(dirh.current_path().value() / "adir", ec));
  // And the temp file must have been cleaned up
  BOOST_CHECK(count_random_files(dirh.current_path().value()) == 0);
}

// A concurrent reader must never observe a partially written destination.
static void TestReaderNeverSeesPartial(const llfio::directory_handle &dirh)
{
  const size_t len = 256 * 1024;
  {
    atomic_file f = atomic_file::open(dirh, "concurrent.txt").value();
    std::string s(len, 'A');
    BOOST_CHECK(f.write(buf(s)).value() == len);
    BOOST_CHECK(f.commit());
  }
  std::atomic<bool> stop{false};
  std::atomic<size_t> reads{0};
  std::atomic<bool> torn{false};
  std::thread reader(
  [&]
  {
    while(!stop.load(std::memory_order_acquire))
    {
      auto fh = llfio::file_handle::file(dirh, "concurrent.txt");
      if(!fh)
      {
        continue;
      }
      std::string s(len, '\0');
      auto r = llfio::read(fh.value(), 0, {{reinterpret_cast<llfio::byte *>(s.data()), s.size()}});
      if(!r)
      {
        continue;
      }
      s.resize(r.value());
      if(!s.empty())
      {
        const char c = s[0];
        for(const char x : s)
        {
          if(x != c)
          {
            torn.store(true, std::memory_order_relaxed);
            break;
          }
        }
      }
      reads.fetch_add(1, std::memory_order_relaxed);
    }
  });
  for(unsigned gen = 1; gen <= 200 && !torn.load(std::memory_order_relaxed); gen++)
  {
    atomic_file f = atomic_file::open(dirh, "concurrent.txt").value();
    std::string s(len, static_cast<char>('A' + (gen % 26)));
    BOOST_REQUIRE(f.write(buf(s)));
    BOOST_REQUIRE(f.commit());
  }
  stop.store(true, std::memory_order_release);
  reader.join();
  BOOST_CHECK(!torn.load());
  BOOST_CHECK(reads.load() > 0);
  {
    std::string s = read_file(dirh, "concurrent.txt");
    BOOST_REQUIRE(s.size() == len);
    const char c = s[0];
    for(const char x : s)
    {
      BOOST_REQUIRE(x == c);
    }
  }
}

// The non-atomic fallback must be used when a temp file cannot be created, must write the
// destination directly, and must refuse to pretend cancellation succeeded.
static void TestFallback(const llfio::directory_handle &dirh)
{
  const llfio::filesystem::path dirpath = dirh.current_path().value();
  const llfio::filesystem::path ropath = dirpath / "ro";
  std::error_code ec;
  llfio::filesystem::create_directory(ropath, ec);
  BOOST_REQUIRE(!ec);
  {
    std::ofstream(ropath / "target.txt") << "original";
  }
  // Make the directory unwritable, but leave the file writable.
  llfio::filesystem::permissions(ropath, llfio::filesystem::perms::owner_read | llfio::filesystem::perms::owner_exec,
                                 llfio::filesystem::perm_options::replace, ec);
  BOOST_REQUIRE(!ec);
  auto rodirh = llfio::directory_handle::directory(dirh, "ro", llfio::file_handle::mode::read).value();
  {
    // Without fallback, this must fail because the temporary file cannot be created.
    auto f = atomic_file::open(rodirh, "target.txt");
    BOOST_REQUIRE(f.has_error());
    // With fallback, it must succeed, and report that it is in fallback mode.
    atomic_file g = atomic_file::open(rodirh, "target.txt", atomic_file::options(true)).value();
    BOOST_CHECK(g.is_fallback());
    // Fallback opens the destination for direct writing and truncates it immediately, so the
    // destination has already been modified before a single byte is written, and the previous
    // contents are already gone. replacement_performed() must report that honestly rather than
    // claiming the destination is untouched because commit() has not run yet.
    BOOST_CHECK(g.replacement_performed());
    BOOST_CHECK(read_file(rodirh, "target.txt").empty());
    BOOST_CHECK(g.write(buf("replaced")).value() == 8);
    // In fallback mode, cancellation cannot be implemented honestly.
    auto c = g.cancel();
    BOOST_REQUIRE(c.has_error());
    BOOST_CHECK(c.error() == llfio::errc::not_supported);
    // Cancellation is refused and the state is unchanged, so commit still works.
    BOOST_CHECK(g.commit());
    BOOST_CHECK(g.current_state() == atomic_file::state::committed);
    BOOST_CHECK(g.replacement_performed());
  }
  BOOST_CHECK(read_file(rodirh, "target.txt") == "replaced");
  // Restore permissions so cleanup can proceed.
  llfio::filesystem::permissions(ropath, llfio::filesystem::perms::owner_all, llfio::filesystem::perm_options::replace,
                                 ec);
}

// Coverage for a failure part way through fallback setup: `open()` has already taken the real
// destination handle but cannot truncate it, so it returns an error and destroys the partially
// constructed transaction.
//
// Honest scope of this test: the only practical way to force the fallback path is a directory we
// can neither create a temporary file in nor delete from, because both need write permission on the
// same directory. So the directory permissions which induce the failure also make the destructive
// `unlink()` this bug is about fail with EACCES, which means this test cannot distinguish correct
// code from the buggy code - it would have passed before the fix too. It is retained as coverage of
// the error path (and asserts we really do reach the truncation failure rather than bailing out
// earlier), while the guarantee itself comes from the structure of the class: the destructor can
// only ever unlink `_tempfile`, and `_tempfile` is never the destination.
#ifndef _WIN32
static void TestFallbackSetupFailure(const llfio::directory_handle &dirh)
{
  const llfio::filesystem::path dirpath = dirh.current_path().value();
  const llfio::filesystem::path ropath = dirpath / "ro2";
  std::error_code ec;
  llfio::filesystem::create_directory(ropath, ec);
  BOOST_REQUIRE(!ec);
  // A destination which can be opened for writing but cannot be truncated: a symlink to the null
  // device, since truncating a character device fails with EINVAL.
  llfio::filesystem::create_symlink("/dev/null", ropath / "target.txt", ec);
  BOOST_REQUIRE(!ec);
  // Make the directory unwritable, so creating a temporary file in it must fail and the fallback
  // path is taken.
  llfio::filesystem::permissions(ropath, llfio::filesystem::perms::owner_read | llfio::filesystem::perms::owner_exec,
                                 llfio::filesystem::perm_options::replace, ec);
  BOOST_REQUIRE(!ec);
  auto rodirh = llfio::directory_handle::directory(dirh, "ro2", llfio::file_handle::mode::read).value();
  {
    auto f = atomic_file::open(rodirh, "target.txt", atomic_file::options(true));
    BOOST_REQUIRE(f.has_error());
    // The failure must come from the truncation, not from an earlier step, otherwise this is not
    // exercising the window this test exists for.
    BOOST_CHECK(f.error() == llfio::errc::invalid_argument);
    // The local transaction is destroyed here: fallback mode, setup failed after taking the
    // destination handle, `_destination_modified` still false. It must not delete the destination.
  }
  ec.clear();
  const llfio::filesystem::file_status st = llfio::filesystem::symlink_status(ropath / "target.txt", ec);
  BOOST_REQUIRE(!ec);
  BOOST_CHECK(llfio::filesystem::is_symlink(st));
  // Restore permissions so cleanup can proceed.
  llfio::filesystem::permissions(ropath, llfio::filesystem::perms::owner_all, llfio::filesystem::perm_options::replace,
                                 ec);
}
#endif

// Move construction and move assignment must transfer ownership of the transaction.
static void TestMove(const llfio::directory_handle &dirh)
{
  {
    atomic_file a = atomic_file::open(dirh, "moved.txt").value();
    BOOST_CHECK(a.write(buf("abc")).value() == 3);
    atomic_file b = std::move(a);
    BOOST_CHECK(b.write(buf("def")).value() == 3);
    BOOST_CHECK(b.commit());
  }
  BOOST_CHECK(read_file(dirh, "moved.txt") == "abcdef");
  {
    atomic_file a = atomic_file::open(dirh, "moved2.txt").value();
    BOOST_CHECK(a.write(buf("ghi")).value() == 3);
    atomic_file b = atomic_file::open(dirh, "moved3.txt").value();
    b = std::move(a);
    BOOST_CHECK(b.commit());
  }
  BOOST_CHECK(read_file(dirh, "moved2.txt") == "ghi");
  // moved3 must not exist, and its temp must have been cleaned up by the move assignment
  BOOST_CHECK(!llfio::file_handle::file(dirh, "moved3.txt"));
  BOOST_CHECK(count_random_files(dirh.current_path().value()) == 0);
}

static inline void TestAtomicFile()
{
  const llfio::filesystem::path dirpath = llfio::filesystem::temp_directory_path() / "llfio_atomic_file_test";
  std::error_code ec;
  llfio::filesystem::remove_all(dirpath, ec);
  BOOST_REQUIRE(llfio::filesystem::create_directory(dirpath, ec));
  BOOST_REQUIRE(!ec);
  {
    auto dirh = llfio::directory_handle::directory({}, dirpath).value();

    TestCreateNewFile(dirh);
    TestOverwriteExisting(dirh);
    TestMultipleWrites(dirh);
    TestEmptyWrite(dirh);
    TestDiscardOnDestruction(dirh);
    TestCancel(dirh);
    TestDoubleCommit(dirh);
    TestPathWithDirectory(dirh);
    TestOpenIntoNonexistentDirectory(dirh);
    TestDestinationIsDirectory(dirh);
    TestReaderNeverSeesPartial(dirh);
    TestFallback(dirh);
#ifndef _WIN32
    TestFallbackSetupFailure(dirh);
#endif
    TestMove(dirh);
  }
  llfio::filesystem::remove_all(dirpath, ec);
}

KERNELTEST_TEST_KERNEL(integration, llfio, atomic_file, atomic_file, "Tests that llfio::atomic_file works as expected",
                       TestAtomicFile())
