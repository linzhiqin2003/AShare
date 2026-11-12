//! Content identity is independent of build directory, compiler cache and date.
use std::{env, process::Command};
fn main() {
    let python = env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".into());
    let script = r#"
import hashlib, pathlib, tomllib
root=pathlib.Path.cwd()
manifest=tomllib.loads((root/'Cargo.toml').read_text())
core=(root/manifest['dependencies']['panelops-core']['path']).resolve()
# Maturin places Python sources at the sdist root, while Cargo's manifest
# remains in packages/panelops. Locate that source root, never omit Python.
python_root=next((base/'python' for base in (root,*root.parents)
                  if (base/'python/panelops/api.py').is_file()),None)
assert python_root is not None, 'panelops Python sources are required for build identity'
files={}
for prefix,base,patterns in [('core',core,['src/**/*.rs']),
                            ('binding',root,['src/**/*.rs','build.rs']),
                            ('binding/python',python_root,['**/*.py'])]:
    for pattern in patterns:
        for p in base.glob(pattern):
            files[prefix+'/'+str(p.relative_to(base))]=p
h=hashlib.sha256()
for name,p in sorted(files.items()):
    data=p.read_bytes()
    h.update(name.encode()+b'\0'+len(data).to_bytes(8,'big')+data)
    print('cargo:rerun-if-changed='+str(p))
print('cargo:rerun-if-changed='+str(python_root))
print('cargo:rustc-env=PANELOPS_BUILD_ID='+h.hexdigest())
"#;
    let output = Command::new(python)
        .args(["-c", script])
        .output()
        .expect("Python required to fingerprint sources");
    assert!(
        output.status.success(),
        "source fingerprint failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    print!(
        "{}",
        String::from_utf8(output.stdout).expect("UTF-8 fingerprint")
    );
    println!("cargo:rerun-if-changed=python");
}
