# Tutor runtime prompt releases

This directory is the small, versioned prompt export used by the Compose tutor. The full Prompt Garden workspace, experiments, and run data stay outside the Docker image.

Each manifest entry binds a stable prompt ID and semantic version to an asset path and SHA-256 digest. Runtime loading verifies the digest before the tutor starts using the prompt. A mismatch stops prompt resolution instead of silently using changed text.

Published prompt assets are immutable. To change a prompt, add a new prompt ID/version and asset, update the manifest's role composition, and retain the previous asset so old invocation records remain resolvable. Do not replace text under an already published ID/version. The manifest currently assigns the baseline; experiment support is represented in the record format, while experiment selection and UI are outside this Change.

Invocation records store only prompt ID, version, digest, component role, model role, provider/model, attempt, and baseline/experiment assignment. They do not store expanded messages, student text, or retrieved document excerpts. Document context and session data can affect a response; their static instructions are separately versioned, while dynamic values are not copied into this provenance table.
