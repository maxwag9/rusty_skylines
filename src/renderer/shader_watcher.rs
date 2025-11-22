use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};

pub struct ShaderWatcher {
    rx: Receiver<notify::Result<Event>>,
    #[allow(dead_code)]
    watcher: RecommendedWatcher,
}

impl ShaderWatcher {
    pub fn new(shader_dir: &PathBuf) -> anyhow::Result<Self> {
        let (tx, rx) = channel();

        let mut watcher = notify::recommended_watcher(move |res| {
            let _ = tx.send(res);
        })?;

        watcher.configure(Config::OngoingEvents(Some(
            std::time::Duration::from_millis(50),
        )))?;
        watcher.watch(shader_dir, RecursiveMode::Recursive)?;

        Ok(Self { rx, watcher })
    }

    pub fn take_changed_paths(&self) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        while let Ok(event) = self.rx.try_recv() {
            if let Ok(ev) = event {
                if matches!(ev.kind, EventKind::Create(_) | EventKind::Modify(_)) {
                    paths.extend(ev.paths);
                }
            }
        }
        paths
    }
}
