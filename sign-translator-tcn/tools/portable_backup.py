"""Create a local transfer archive, or restore into a new empty directory."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'yolo-runtime', 'yolo26-runtime', 'yolo-config', '__pycache__', '.venv', '.git'}
SOURCE_SUFFIXES = {'.py', '.js', '.html', '.css', '.md', '.txt', '.ps1', '.bat'}


def selected_files(root):
    for path in root.iterdir():
        if path.is_file() and (path.suffix in SOURCE_SUFFIXES or path.name in ('package.json', '.gitignore')):
            yield path
    for folder in ('frontend', 'backend', 'training', 'tools', 'scripts', 'docs', 'tests', 'vendor', 'data'):
        for path in (root / folder).rglob('*'):
            relative = path.relative_to(root)
            if path.is_file() and not path.is_symlink() and not set(relative.parts) & EXCLUDED and path.suffix not in ('.log', '.pyc', '.tmp'):
                yield path


def backup(root, output):
    hashes = {}
    # Exclusive creation prevents accidental replacement of an existing backup.
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in selected_files(root):
            name = path.relative_to(root).as_posix()
            digest = hashlib.sha256()
            with path.open('rb') as source, archive.open(name, 'w', force_zip64=True) as destination:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    destination.write(chunk)
            hashes[name] = digest.hexdigest()
        archive.writestr('transfer-manifest.json', json.dumps(hashes, ensure_ascii=False))
    return len(hashes)


def restore(archive_path, destination):
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Restore destination must be empty. Existing data will not be overwritten.')
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive paths')
        for entry in entries:
            name = PurePosixPath(entry.filename)
            target = destination.joinpath(*name.parts).resolve()
            if name.is_absolute() or '..' in name.parts or ':' in entry.filename or '\\' in entry.filename or not target.is_relative_to(destination) or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError('Unsafe archive path')
        manifest = json.loads(archive.read('transfer-manifest.json'))
        if set(manifest) != set(names) - {'transfer-manifest.json'}:
            raise ValueError('Invalid transfer manifest')
        # Validate all bytes before writing any restored file.
        for name, expected in manifest.items():
            digest = hashlib.sha256()
            with archive.open(name) as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError('Backup checksum mismatch: ' + name)
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination)
    return len(manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('archive', type=Path)
    load = commands.add_parser('restore')
    load.add_argument('archive', type=Path)
    load.add_argument('destination', type=Path)
    args = parser.parse_args()
    if args.command == 'export':
        count = backup(ROOT, args.archive)
    else:
        count = restore(args.archive, args.destination)
    print(f'Completed: {count} files. Personal training data is private; do not upload publicly.')
