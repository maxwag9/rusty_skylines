use notify::{Event, EventKind, RecursiveMode, Watcher, recommended_watcher};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};

pub struct ShaderWatcher {
    pub rx: Receiver<notify::Result<Event>>,
    _watcher: notify::RecommendedWatcher,
    watched_files: HashSet<PathBuf>,
    watched_dirs: HashSet<PathBuf>,
}

impl ShaderWatcher {
    pub fn new(paths: Vec<PathBuf>) -> anyhow::Result<Self> {
        let (tx, rx) = channel();

        let watcher = recommended_watcher(move |res| {
            let _ = tx.send(res);
        })?;

        let mut this = Self {
            rx,
            _watcher: watcher,
            watched_files: HashSet::new(),
            watched_dirs: HashSet::new(),
        };

        this.update(paths)?;

        Ok(this)
    }

    pub fn update(&mut self, paths: Vec<PathBuf>) -> anyhow::Result<()> {
        let new_files: HashSet<PathBuf> = paths.into_iter().collect();

        let new_dirs: HashSet<PathBuf> = new_files
            .iter()
            .filter_map(|path| path.parent().map(PathBuf::from))
            .collect();

        for dir in self.watched_dirs.difference(&new_dirs) {
            self._watcher.unwatch(dir)?;
        }

        for dir in new_dirs.difference(&self.watched_dirs) {
            self._watcher.watch(dir, RecursiveMode::NonRecursive)?;
        }

        self.watched_files = new_files;
        self.watched_dirs = new_dirs;

        while self.rx.try_recv().is_ok() {}

        Ok(())
    }

    pub fn take_changed_wgsl_files(&self) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();

        while let Ok(Ok(event)) = self.rx.try_recv() {
            if !matches!(
                event.kind,
                EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
            ) {
                continue;
            }

            for path in event.paths {
                if self.watched_files.contains(&path) && seen.insert(path.clone()) {
                    out.push(path);
                }
            }
        }

        out
    }
}
