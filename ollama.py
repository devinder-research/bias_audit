"""Standard-library transport, immutable identities, and crash-safe response cache."""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def http(url, payload=None, timeout=300):
    body = None if payload is None else canonical(payload).encode()
    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)


def preflight(config, frozen=None):
    base = config['ollama_url']
    tags = {m['name']: m['digest'] for m in http(base + '/api/tags')['models']}
    models = {}
    for tag, prefix in config['models'].items():
        digest = tags.get(tag, '')
        if not digest.startswith(prefix):
            raise ValueError(f'Missing or unexpected model {tag}: {digest!r}; never pulling models.')
        models[tag] = digest
    state = {'models': models, 'ollama_version': http(base + '/api/version')['version']}
    if frozen and state != frozen:
        raise ValueError('Ollama model digest or version changed from frozen manifest.')
    return state


def unload(config, model):
    http(config['ollama_url'] + '/api/chat', {'model': model, 'messages': [], 'keep_alive': 0})


class Cache:
    def __init__(self, path, protocol):
        self.path = Path(path)
        self.protocol = protocol
        self.records = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            # A killed append may leave only the final line incomplete; discard that tail.
            with self.path.open('rb+') as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith(b'\n'):
                        handle.truncate(offset)
                        break
                    row = json.loads(line)
                    checksum = row.pop('checksum', None)
                    if checksum != fingerprint(row):
                        raise ValueError('Cache record checksum mismatch.')
                    row['checksum'] = checksum
                    if row['protocol'] != protocol or row['key'] != fingerprint(row['request']):
                        raise ValueError('Cache identity/protocol mismatch.')
                    if row['key'] in self.records:
                        raise ValueError('Duplicate cache request.')
                    self.records[row['key']] = row

    def get(self, request):
        return self.records.get(fingerprint(request))

    def add(self, request, raw, parsed, elapsed, response=None):
        key = fingerprint(request)
        if key in self.records:
            raise ValueError('Attempt to append a duplicate request.')
        row = {'key': key, 'protocol': self.protocol, 'request': request,
               'raw': raw, 'parsed': parsed, 'elapsed_seconds': elapsed,
               'response': response or {}}
        row['checksum'] = fingerprint(row)
        with self.path.open('a') as handle:
            handle.write(canonical(row) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        self.records[key] = row
        return row

    def run(self, config, request, case):
        existing = self.get(request)
        if existing is not None:
            return existing
        from prompts import parse_choice
        start = time.monotonic()
        response = http(config['ollama_url'] + '/api/chat', {
            'model': request['model'], 'messages': request['messages'], 'stream': False,
            'options': request['options'], 'keep_alive': '10m',
        })
        raw = response['message']['content']
        parsed = parse_choice(raw, case['answers']) if case['kind'] == 'bbq' else None
        return self.add(request, raw, parsed, time.monotonic() - start, response)
