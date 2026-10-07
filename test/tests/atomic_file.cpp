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

#ifndef _WIN32
// For `mode_t` and the process umask, used by the permission preservation test.
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef __linux__
// For the private mount namespace in which the fallback setup failure test runs.
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sched.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

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

#ifndef _WIN32
// The full path of a leaf within the test directory.
static std::string path_of(const llfio::directory_handle &dirh, llfio::path_view p)
{
  return (dirh.current_path().value() / p.path()).string();
}

// Reads a path's permission bits with a plain POSIX stat, deliberately independently of the
// mechanism `atomic_file` uses, so that this checks the observable outcome rather than repeating the
// implementation's own logic. Note this needs no permission on the file itself, only on its parent.
static uint16_t perms_of(const llfio::directory_handle &dirh, llfio::path_view p)
{
  const std::string full = path_of(dirh, p);
  struct stat s
  {
  };
  BOOST_REQUIRE(::stat(full.c_str(), &s) == 0);
  return static_cast<uint16_t>(s.st_mode & 0xfff);
}

// Creates a file if needed, then sets its permission bits with a plain POSIX chmod.
static bool set_perms(const llfio::directory_handle &dirh, llfio::path_view p, uint16_t mode)
{
  auto fh = llfio::file_handle::file(dirh, p, llfio::file_handle::mode::write, llfio::file_handle::creation::if_needed);
  if(!fh)
  {
    return false;
  }
  BOOST_REQUIRE(fh.value().close());
  return ::chmod(path_of(dirh, p).c_str(), static_cast<mode_t>(mode)) == 0;
}

// Replacing an existing file must leave its permission bits alone, as QSaveFile does. Without that
// an atomic save silently changes the mode of the destination: it drops the executable bit off a
// script, and it widens a file the caller had tightened.
static void TestPreservePermissions(const llfio::directory_handle &dirh)
{
  /* Read the process umask using the standard round trip. A temporary file is created with mode
  0660 *before* the umask is applied, so that is the mode a replaced destination would be left with
  were its own permissions not preserved. Every mode tested below is unequal to it for any umask, so
  a failure here means a failure to preserve, and not something else.
  */
  const mode_t um = ::umask(0);
  ::umask(um);
  const uint16_t unpreserved = static_cast<uint16_t>(0660 & ~um);
  // 0644 and 0755 set "other" bits, and 0700 sets an owner execute bit, none of which the creation
  // mode 0660 can ever set. 0600 and 0640 are the realistic tightened modes.
  const uint16_t modes[] = {0600, 0640, 0644, 0700, 0755};
  // Precondition: this filing system must be able to report a mode at all. Note this is a skip, not
  // a failure: a filing system with no permissions cannot have lost any.
  if(!set_perms(dirh, "perms_probe.txt", 0644) || perms_of(dirh, "perms_probe.txt") != 0644)
  {
    BOOST_TEST_MESSAGE("this filing system does not support permissions, skipping");
    return;
  }
  size_t index = 0;
  for(const uint16_t mode : modes)
  {
    const std::string name = "perms" + std::to_string(index++) + ".txt";
    if(mode == unpreserved)
    {
      continue;  // this one would be preserved by accident, so it could not detect a failure
    }
    BOOST_REQUIRE(set_perms(dirh, name, mode));
    {
      atomic_file f = atomic_file::open(dirh, name).value();
      BOOST_CHECK(f.write(buf("contents")).value() == 8);
      BOOST_CHECK(f.commit());
    }
    BOOST_CHECK(read_file(dirh, name) == "contents");
    const uint16_t got = perms_of(dirh, name);
    if(got != mode)
    {
      /* NOTE: in this test framework BOOST_CHECK_MESSAGE only prints, it does not register a
      failure. So print the diagnosis here, and register the failure with a plain BOOST_CHECK. */
      BOOST_TEST_MESSAGE("destination mode " << std::oct << mode << " became " << std::oct << got);
    }
    BOOST_CHECK(got == mode);
  }
  {
    // A destination which the caller cannot even read must still keep its mode: it is read through
    // an anchor handle, which needs no permission on the destination itself. Its contents cannot be
    // read back here for that same reason, so this part checks the mode alone.
    BOOST_REQUIRE(set_perms(dirh, "perms0000.txt", 0000));
    BOOST_CHECK_EQUAL(perms_of(dirh, "perms0000.txt"), 0000);
    atomic_file f = atomic_file::open(dirh, "perms0000.txt").value();
    BOOST_CHECK(f.write(buf("contents")).value() == 8);
    BOOST_CHECK(f.commit());
    BOOST_CHECK_EQUAL(perms_of(dirh, "perms0000.txt"), 0000);
  }
  {
    // A destination which does not exist has no mode to preserve, so the new file keeps the mode the
    // temporary file was created with. This feature deliberately does not invent permissions for a
    // file which was not there before.
    atomic_file f = atomic_file::open(dirh, "perms_new.txt").value();
    BOOST_CHECK(f.write(buf("fresh")).value() == 5);
    BOOST_CHECK(f.commit());
    BOOST_CHECK(read_file(dirh, "perms_new.txt") == "fresh");
    BOOST_CHECK_EQUAL(perms_of(dirh, "perms_new.txt"), unpreserved);
  }
}
#endif

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

// Regression test for a fallback-mode transaction deleting the caller's destination.
//
// The scenario is narrow and needs the right conditions to be observable at all: `open()` must fail
// part way through fallback setup, after it has taken a handle to the real destination but before
// it has recorded that the destination was modified. The partially constructed transaction is then
// destroyed, and its destructor must not unlink the destination.
//
// Choosing those conditions takes care, because the obvious arrangements hide the bug rather than
// expose it. Fallback is reached by making the temporary file uncreatable, so start by making the
// directory unwritable: the destructive unlink then fails with EACCES on that same directory, and
// the test passes whether or not the bug is present. Instead point the destination at a symlink to
// the null device - opening it for writing succeeds, `ftruncate` fails - but that hides the bug too,
// because the entry and the handle then disagree and LLFIO's safety unlink refuses to remove it.
//
// So the destination has to be an entry which opens for writing, cannot be truncated, and *is* the
// object the handle refers to. A FIFO is all three, provided its read end is held open. And fallback
// is triggered without touching permissions by exhausting the filesystem's inodes: creating a new
// file then fails, while deleting an existing one still succeeds, so a destructor which unlinks the
// destination really does delete it. Exhausting inodes needs a filesystem we control, so this runs
// in a child process inside a private mount namespace with its own tiny tmpfs, destroyed along with
// the child. That needs root or unprivileged user namespaces; otherwise the test skips rather than
// reporting a result it did not obtain.
#ifndef __linux__
static void TestFallbackSetupFailure(const llfio::directory_handle &)
{
  BOOST_TEST_MESSAGE(
  "Fallback setup failure test needs tmpfs and mount namespaces, which are Linux only. So skipping this test.");
}
#else
// Write a small string to a pseudo-file, used to configure user namespace id maps.
static bool write_pseudofile(const char *path, const char *contents)
{
  const int fd = ::open(path, O_WRONLY);
  if(fd < 0)
  {
    return false;
  }
  const size_t len = ::strlen(contents);
  const bool ok = ::write(fd, contents, len) == (ssize_t) len;
  ::close(fd);
  return ok;
}

// Runs in the child process. Returns 0 if the destination survived, 77 if the environment cannot
// host the test, and any other value to identify which step failed. Deliberately uses no assertion
// macro: the parent asserts on the exit code.
static int fallback_setup_failure_child(int userns_ready_fd, int userns_go_fd, const char *mountpoint)
{
  if(::geteuid() != 0)
  {
    // Become root inside a new user namespace, which is enough to mount tmpfs.
    if(::unshare(CLONE_NEWUSER) != 0)
    {
      return 77;
    }
    // The id maps have to be written by some *other* process: having just called unshare(CLONE_NEWUSER)
    // we no longer hold the capability to write our own. Hand over to the parent and wait for it.
    if(::write(userns_ready_fd, "1", 1) != 1)
    {
      return 77;
    }
    char go = 0;
    if(::read(userns_go_fd, &go, 1) != 1)
    {
      return 77;
    }
  }
  if(::unshare(CLONE_NEWNS) != 0)
  {
    return 77;
  }
  // Stop the mount below propagating back into the parent's namespace.
  (void) ::mount("none", "/", nullptr, MS_REC | MS_PRIVATE, nullptr);
  if(::mount("tmpfs", mountpoint, "tmpfs", 0, "size=4M,nr_inodes=8") != 0)
  {
    return 77;
  }
  const std::string sub = std::string(mountpoint) + "/sub";
  if(::mkdir(sub.c_str(), 0755) != 0)
  {
    return 1;
  }
  const std::string target = sub + "/target.txt";
  if(::mkfifo(target.c_str(), 0644) != 0)
  {
    return 1;
  }
  // Hold the read end open: that is what stops opening the FIFO for writing from blocking.
  const int fifo_reader = ::open(target.c_str(), O_RDONLY | O_NONBLOCK);
  if(fifo_reader < 0)
  {
    return 1;
  }
  // Use up every remaining inode, so that creating a new file must now fail.
  int fillers = 0;
  for(; fillers < 64; fillers++)
  {
    const std::string filler = sub + "/filler" + std::to_string(fillers);
    const int fd = ::open(filler.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
    if(fd < 0)
    {
      break;
    }
    ::close(fd);
  }
  if(fillers == 0 || fillers == 64)
  {
    return 77;  // inodes not exhausted, or not exhausted within a sane number of files
  }
  // The precondition which makes the bug observable: deleting must still be possible. If it is not,
  // a destructor unlinking the destination would fail harmlessly and this test would prove nothing.
  if(::unlink((sub + "/filler0").c_str()) != 0)
  {
    return 77;
  }
  const int refill = ::open((sub + "/filler0").c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
  if(refill < 0)
  {
    return 77;
  }
  ::close(refill);
  {
    const int probe = ::open((sub + "/probe").c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
    if(probe >= 0)
    {
      ::close(probe);
      (void) ::unlink((sub + "/probe").c_str());
      return 77;  // creating a file still works, so fallback would not be reached
    }
  }
  {
    auto dirh = llfio::directory_handle::directory({}, sub);
    if(!dirh)
    {
      return 2;
    }
    {
      auto f = atomic_file::open(dirh.value(), "target.txt", atomic_file::options(true));
      if(!f.has_error())
      {
        return 3;  // the destination cannot be truncated, so this must fail
      }
      if(f.error() != llfio::errc::invalid_argument)
      {
        return 4;  // and it must be the truncation failing, not some earlier step
      }
      // `open()` destroys the partially constructed transaction before returning to us.
    }
    ::close(fifo_reader);
    struct stat st{};
    if(::lstat(target.c_str(), &st) != 0)
    {
      return 5;  // the destination is gone, which is the bug
    }
    if(!S_ISFIFO(st.st_mode))
    {
      return 6;
    }
  }
  return 0;
}

static void TestFallbackSetupFailure(const llfio::directory_handle &dirh)
{
  const llfio::filesystem::path mountpoint = dirh.current_path().value() / "fallback_ns";
  std::error_code ec;
  llfio::filesystem::create_directory(mountpoint, ec);
  BOOST_REQUIRE(!ec);
  const std::string mountpoint_str = mountpoint.string();
  const bool need_userns = ::geteuid() != 0;
  // Writing to a pipe whose reader has already gone would raise SIGPIPE, which must not abort the run.
  (void) ::signal(SIGPIPE, SIG_IGN);
  int userns_ready[2] = {-1, -1};
  int userns_go[2] = {-1, -1};
  BOOST_REQUIRE(::pipe(userns_ready) == 0);
  BOOST_REQUIRE(::pipe(userns_go) == 0);
  const pid_t pid = ::fork();
  BOOST_REQUIRE(pid >= 0);
  if(pid == 0)
  {
    ::close(userns_ready[0]);
    ::close(userns_go[1]);
    ::_exit(fallback_setup_failure_child(userns_ready[1], userns_go[0], mountpoint_str.c_str()));
  }
  ::close(userns_ready[1]);
  ::close(userns_go[0]);
  bool maps_written = false;
  if(need_userns)
  {
    char ready = 0;
    // Only write the id maps, and only release the child, if it actually got as far as asking.
    if(::read(userns_ready[0], &ready, 1) == 1)
    {
      char procpath[64];
      char map[64];
      ::snprintf(procpath, sizeof(procpath), "/proc/%d/setgroups", (int) pid);
      (void) write_pseudofile(procpath, "deny");
      ::snprintf(map, sizeof(map), "0 %u 1", (unsigned) ::getuid());
      ::snprintf(procpath, sizeof(procpath), "/proc/%d/uid_map", (int) pid);
      maps_written = write_pseudofile(procpath, map);
      if(maps_written)
      {
        ::snprintf(map, sizeof(map), "0 %u 1", (unsigned) ::getgid());
        ::snprintf(procpath, sizeof(procpath), "/proc/%d/gid_map", (int) pid);
        maps_written = write_pseudofile(procpath, map);
      }
      (void) ::write(userns_go[1], "1", 1);
    }
  }
  ::close(userns_ready[0]);
  ::close(userns_go[1]);
  int status = 0;
  BOOST_REQUIRE(::waitpid(pid, &status, 0) == pid);
  BOOST_REQUIRE(WIFEXITED(status));
  const int code = WEXITSTATUS(status);
  if(code == 77 || (need_userns && !maps_written))
  {
    BOOST_TEST_MESSAGE(
    "Fallback setup failure test needs root or usable unprivileged user namespaces. So skipping this test.");
    return;
  }
  if(code != 0)
  {
    // Note: BOOST_CHECK_MESSAGE() in this framework only logs, it does not register a failure, so the
    // diagnosis goes out as a message and the actual pass/fail goes through BOOST_CHECK().
    BOOST_TEST_MESSAGE("fallback setup failure test: child process exited " << code
                                                                            << " (5 means it deleted the destination)");
  }
  BOOST_CHECK(code == 0);
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
#ifndef _WIN32
    TestPreservePermissions(dirh);
#endif
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
    TestFallbackSetupFailure(dirh);
    TestMove(dirh);
  }
  llfio::filesystem::remove_all(dirpath, ec);
}

KERNELTEST_TEST_KERNEL(integration, llfio, atomic_file, atomic_file, "Tests that llfio::atomic_file works as expected",
                       TestAtomicFile())
