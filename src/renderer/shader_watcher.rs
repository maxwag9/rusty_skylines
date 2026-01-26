use crate::paths::shader_dir;
use notify::{Event, EventKind, RecursiveMode, Watcher, recommended_watcher};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};

pub struct ShaderWatcher {
    pub rx: Receiver<notify::Result<Event>>,
    _watcher: notify::RecommendedWatcher,
}

impl ShaderWatcher {
    pub fn new() -> anyhow::Result<Self> {
        let (tx, rx) = channel();

        let mut watcher = recommended_watcher(move |res| {
            let _ = tx.send(res);
        })?;

        watcher.watch(shader_dir().as_path(), RecursiveMode::Recursive)?;

        Ok(Self {
            rx,
            _watcher: watcher,
        })
    }

    pub fn take_changed_wgsl_files(&self) -> Vec<PathBuf> {
        let mut out = Vec::new();
        while let Ok(Ok(event)) = self.rx.try_recv() {
            if matches!(
                event.kind,
                EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
            ) {
                out.extend(
                    event
                        .paths
                        .into_iter()
                        .filter(|path| path.extension().and_then(|x| x.to_str()) == Some("wgsl")),
                );
            }
        }

        out
    }
}
