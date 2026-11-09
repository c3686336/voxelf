use std::ffi::OsString;
use std::path::PathBuf;

pub struct Options {
    pub environment: PathBuf,
    pub capture: Option<PathBuf>,
    pub samples: Option<u32>,
    pub renderer: Option<Renderer>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Renderer {
    Reference,
    Albedo,
}

impl Options {
    pub fn parse(args: impl IntoIterator<Item = OsString>) -> anyhow::Result<Self> {
        let mut args = args.into_iter();
        let mut environment = None;
        let mut capture = None;
        let mut samples = None;
        let mut renderer = None;
        while let Some(arg) = args.next() {
            match arg.to_str() {
                Some("--capture") => {
                    anyhow::ensure!(capture.is_none(), "--capture specified more than once");
                    capture = Some(PathBuf::from(args.next().ok_or_else(|| {
                        anyhow::anyhow!("--capture requires an output .hdr or .png path")
                    })?));
                }
                Some("--samples") => {
                    anyhow::ensure!(samples.is_none(), "--samples specified more than once");
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--samples requires an integer"))?;
                    let value = value
                        .to_str()
                        .ok_or_else(|| anyhow::anyhow!("--samples must be UTF-8"))?
                        .parse::<u32>()?;
                    anyhow::ensure!(
                        (1..=1024).contains(&value),
                        "--samples must be between 1 and 1024"
                    );
                    samples = Some(value);
                }
                Some("--renderer") => {
                    anyhow::ensure!(renderer.is_none(), "--renderer specified more than once");
                    renderer = Some(
                        match args.next().and_then(|s| s.into_string().ok()).as_deref() {
                            Some("reference") => Renderer::Reference,
                            Some("albedo") => Renderer::Albedo,
                            _ => anyhow::bail!("--renderer must be reference or albedo"),
                        },
                    );
                }
                Some(value) if value.starts_with('-') => anyhow::bail!("unknown option: {value}"),
                _ => {
                    anyhow::ensure!(
                        environment.is_none(),
                        "only one environment path may be specified"
                    );
                    environment = Some(PathBuf::from(arg));
                }
            }
        }
        if let Some(path) = &capture {
            let extension = path
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("");
            anyhow::ensure!(
                extension.eq_ignore_ascii_case("hdr") || extension.eq_ignore_ascii_case("png"),
                "--capture output must end in .hdr or .png"
            );
        }
        Ok(Self {
            environment: environment.unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../res/envmap.hdr")
            }),
            capture,
            samples,
            renderer,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_capture_options_and_rejects_invalid_sample_counts() {
        let options = Options::parse(
            [
                "--capture",
                "frame.png",
                "--samples",
                "128",
                "--renderer",
                "albedo",
                "env.hdr",
            ]
            .map(OsString::from),
        )
        .unwrap();
        assert_eq!(options.capture, Some(PathBuf::from("frame.png")));
        assert_eq!(options.environment, PathBuf::from("env.hdr"));
        assert_eq!(options.samples, Some(128));
        assert_eq!(options.renderer, Some(Renderer::Albedo));
        for value in ["0", "1025", "not-a-number"] {
            assert!(Options::parse(["--samples", value].map(OsString::from)).is_err());
        }
    }
}
