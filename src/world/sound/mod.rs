pub mod car_sounds;
pub mod sfx;
pub mod sound;

pub const MAX_CARS_AUDIO: usize = 10;

fn with_stderr_suppressed<T>(f: impl FnOnce() -> T) -> T {
    #[cfg(unix)]
    unsafe {
        use std::os::unix::io::AsRawFd;

        let stderr_fd = libc::dup(libc::STDERR_FILENO);
        let devnull = std::fs::OpenOptions::new()
            .write(true)
            .open("/dev/null")
            .unwrap();

        libc::dup2(devnull.as_raw_fd(), libc::STDERR_FILENO);
        let out = f();
        libc::dup2(stderr_fd, libc::STDERR_FILENO);
        libc::close(stderr_fd);
        out
    }

    #[cfg(not(unix))]
    {
        f()
    }
}
