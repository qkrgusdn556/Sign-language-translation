# Local Recording Contract v1

## Storage and Ownership

New recordings are append-only under `data/recordings/clips/<server-generated-id>/`.
Each directory contains `rgb.webm` and `metadata.json`. The existing `sign-samples.json`,
training snapshots and model files are not modified. New recordings are not automatically
included in legacy TCN training. The directory is ignored by Git and direct static access
is blocked. Playback is available through the local, same-origin recording API.

The user must start the camera, consent to local video/landmark storage, and explicitly
start recording. Recording uses only the camera's video tracks, without an overlay,
mirror transform or audio. The default duration limit is 30 seconds and size limit 32MB.
Stopping the camera also ends the recorder. Failed uploads remain in page memory for
retry; closing the page can lose an unsaved recording.

## Fields

| Field | Meaning |
| --- | --- |
| schema_version | 1 |
| request_id | Client UUID, retry/idempotency token, not a file path |
| clip_id | Server-generated storage ID |
| session_id | Anonymous capture session, ASCII identifier |
| split | train / validation / test; immutable within a session ID |
| kind | isolated / continuous / rest / non_sign / transition |
| sequence | Ordered sign labels; background kinds require an empty list |
| segments | Optional ordered label/start_ms/end_ms ground truth, relative to capture start |
| consent / audio | true / false |
| camera | Actual reported dimensions/FPS and mirrored:false; no device ID |
| started_at_utc | Browser wall-clock start; not an encoded frame timestamp |
| duration_ms | Monotonic recorder start-to-stop elapsed time |
| frames | Live observation events described below |
| predictions | Confirmed output events: label/source/at_ms, optional token key |
| pipeline | TCN/DTW and joint-only/consensus fingerspelling selection |
| context / context_end | Code digest and loaded TCN checkpoint digest at beginning/end |
| client_model | Active sample digest, uploaded YOLO model/label digests, and precision_tracking toggle |
| video_sha256 / video_bytes | Server-computed immutable video fingerprint and size |
| video_validation | header_only_decode_pending |
| synchronization | preview_observations_not_encoded_frame_mapping |
| face_mouth | not_extracted |
| training_eligible | false until decoding, extraction, synchronization and review are implemented |

Every observation includes `elapsed_ms`, camera `media_time_ms`, raw hand landmarks,
raw handedness results, optional world-hand landmarks, and raw Pose landmarks only on
frames where Pose actually ran. `pose_status` is detected/missing/not_run. Hand and Pose
coordinates are recorded before display smoothing. Missing observations are not filled
with the last successful landmarks. Per-hand classification scores are not copied into
per-point confidence fields. Face/mouth landmarks are not produced by this collector.

Optional `frames[].refinement` records a same-frame crop observation separately from
the raw hands: source index, pixel crop/dimensions, elapsed worker wait, mapped image
landmarks, accepted flag and rejection reason. Raw hand coordinates are never replaced
in stored observations. Accepted crop results may feed live recognition after smoothing;
input-activity admission always uses separately smoothed full-frame observations so
switching between crops and the full frame cannot itself arm recognition. The crop is
an additional estimate, not ground truth or reconstruction of an occluded hand.

The observation sequence is NOT the encoded WebM frame sequence. Browser scheduling,
camera timestamps and encoded presentation timestamps can have different origins and
dropped frames. Do not relabel observation order as decoded frame index. Actual WebM
decode validation and frame-synchronized re-extraction remain prerequisites for SLRT
training. Header validation does not prove a fully decodable, audio-free video; audio
is excluded by the browser's video-only recording stream and must be audited on import.

## Split Safety

An existing session ID cannot change split. Identical video bytes cannot be saved twice,
even under another session. These checks do not discover renamed or re-encoded near
duplicates, or two session IDs used for the same physical filming session. Users must
keep filming sessions together and reserve genuinely new sessions for evaluation.
Old samples have no reliable session provenance and are never silently migrated.

## Evaluation

`/api/recordings/evaluation` scores only validation/test recordings. Training recordings
are not used for displayed scores. Results are grouped by split, pipeline, source digest,
loaded model digest, sample digest, and uploaded YOLO identity. Capture runs whose
server model/code context changed are excluded, not pooled into a score.

Sign sequences use minimum-edit alignment with deterministic diagonal/deletion/insertion
tie breaking. WER is total substitutions+deletions+insertions divided by total reference
words, not a per-clip average or a calibrated accuracy. Omission rate is deletions/reference
words. Rest/non_sign/transition false outputs are counted per minute of those recordings.
Only confirmed outputs are evaluated; tentative candidates are not counted as words.
Successive automatic fingerspelling updates of the same token replace its earlier form.

Latency uses aligned exact matches with manually supplied sign-end times. Early outputs
are counted separately; nonnegative browser-clock delays produce nearest-rank p50/p95.
Missing timed labels yield null latency, not zero. This is live-run measurement, not
replay inference on identical video across engines. Decoded-frame timing, recognition
generalization and fair cross-engine replay are not implemented here. Opt-in isolated
and background SLRT import is now implemented separately; see SHARED-LEARNING.md.

## Shared Training Extension

Optional share_slrt:true explicitly opts a new recording into the local native dataset
bridge. /api/shared-motion also carries motion_sample for the same labelled train clip;
successful completion adds that observed 130-value sequence to the active sample file.
The sample file retains applied_shared_motion IDs for idempotent retry. Video collection
does not trigger training. Native model training uses a separate explicit button and
re-extracts synchronized inputs from decoded video rather than trusting preview timing.
