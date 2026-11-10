# Shared Motion Collection

New two-second captures can be recorded once for DTW/TCN and the separate MediaPipe/RGB
SLRT experiment. This is dataset sharing, not full official SLRT or live score fusion.
No old landmark-only sample can supply a missing original video.

## Ownership

Only this application's source is changed. The SLRT project's existing code, videos and
models remain untouched by collection. New native-compatible clips are appended through
the fixed local bridge to its data/clips directory. There are no remote uploads or
client-supplied destination paths. Backend CLI overrides support isolated tests and
disable real SLRT training whenever an alternate import storage is selected.

## Commit Stages

POST /api/shared-motion validates a single train label and a finite 130-value motion
sequence. It first commits the original video through the append-only recording store.
It then atomically appends the motion sample and an applied_shared_motion source-ID list
to sign-samples.json. Ordinary sample edits preserve the source-ID list, preventing a
retry from silently reintroducing a deleted shared sample. Finally, the bridge copies
the original video byte-for-byte into a deterministic native clip directory.

These stores are not one filesystem transaction. An interrupted sample commit retains
the original video; the identical request is safe to retry. An SLRT import failure does
not delete a successful DTW/TCN save. The response explicitly distinguishes that failure,
and /api/slrt/sync retries only recordings with share_slrt:true. No old recording is
implicitly opted into sharing. Hashes, labels, session split and native clip bytes are
checked before accepting an existing import. Conflicts refuse overwrite.

## Native Contract

The native contract validator is loaded read-only from the installed experiment.
The bridge records source clip/context provenance, ordered labels, camera dimensions,
UTC start, duration and real observation clocks. Repeated preview media times are
omitted rather than invented. Copied metadata explicitly says decoding, landmarks and
encoded-frame mapping are pending. Browser filtered 130-value vectors are not presented
as native raw MediaPipe or frame-synchronized RGB features. Native training re-extracts
both inputs from decoded frames using the experiment's existing extractor.

Isolated and background clips are supported. Validation/test clips preserve their split
and the native trainer excludes them. Continuous clips require segment extraction and
are not auto-imported into the whole-clip classifier. Raw recordings and sample edits
remain separate ownership domains; deleting a landmark sample does not delete video.

## Training

An explicit SLRT training button checks the native counts and launches the native
TrainingJob using the installed Python environment, with its existing cross-process
lock, bounded training, model versioning and progress file. It uses all native train
records, including existing ones, and never auto-trains after collection. The TCN train
button remains separate. Status reports pending native manifest entries rather than
claiming collection means training completion or measured accuracy.

Tests use synthetic video and a separate import/sample store. Real training is disabled
there. Functional recording/import tests and mocked subprocess delegation do not establish
model accuracy or successful end-to-end learning on new user recordings.
