# SignLearn – Full-Scale Implementation Plan

This implementation plan translates your project proposal into a structured, actionable roadmap. It is tailored for a two-person final-year team, accounting for the technical scope, stakeholder feedback loops, and the iterative, user-centered methodology outlined in your proposal. The goal is to deliver a functional, well-documented prototype ready for pre-defense and final defense within a 14-week timeframe.

---

## 1. Project Timeline Overview (14 Weeks)

The work is divided into seven sequential phases, with milestones and continuous testing embedded. The timeline assumes both members contribute approximately 15–20 hours per week.

| Phase | Duration | Key Focus |
| --- | --- | --- |
| 0: Project Setup & Environment | Week 1 | Dev environment, tools, repository, initial dataset acquisition |
| 1: Data Pipeline & Preprocessing | Weeks 2–3 | Dataset finalization, landmark extraction pipeline, preprocessing scripts |
| 2: Model Development (Core AI) | Weeks 4–6 | LSTM model design, training, validation, and optimization |
| 3: Backend & API Development | Weeks 5–7 | Flask server, WebSocket integration, model serving |
| 4: Frontend Development | Weeks 6–9 | React UI, webcam capture, speech-to-text, dual-panel layout |
| 5: Full System Integration | Week 10 | End-to-end pipeline connectivity, real-time streaming optimization |
| 6: Testing, Evaluation & Usability | Weeks 11–12 | Performance metrics, formative user testing, bug fixes |
| 7: Documentation & Defense Prep | Weeks 13–14 | Technical report, user manual, demo video, pre-defense rehearsal |

---

## 2. Detailed Phase-Wise Plan

### Phase 0: Project Setup & Environment (Week 1)

**Objective:** Establish all infrastructure, tools, and shared knowledge base so development can proceed without friction.

**Tasks:**

1. **Set up version control:** Create a GitHub repository with `.gitignore` (Python/Node), branches (`main`, `dev-ml`, `dev-web`), and a project board (e.g., GitHub Projects) to track tasks.
2. **Environment configuration:**
    - Create a shared Conda/virtual environment `signlearn-env` with Python 3.9+, install core libraries: TensorFlow/Keras, MediaPipe, OpenCV, NumPy, Pandas.
    - Set up a separate Node.js/React environment for the frontend.
3. **Hardware check:** Test both team members’ webcams (Logitech HD or built-in) and ensure uniform resolution (≥720p, 30fps).
4. **Literature deep-dive:** Review the papers referenced in your proposal (MediaPipe, LSTM, WLASL dataset) and agree on the target ASL vocabulary of ~70 signs (alphabet + numbers + 30–40 common phrases).
5. **Dataset identification:** Download and verify the primary datasets:
    - ASL Alphabet Dataset (Kaggle/Mendeley)
    - ASL Digits (Sign Language MNIST)
    - WLASL or AUTSL subsets for dynamic signs
    - If dynamic video datasets are insufficient, plan to use frame-sequence approximations.
6. **Stakeholder contact:** Reach out to university disability support units or local DHH organizations to schedule 2–3 consultation sessions (tentatively Weeks 7, 10, 12).

**Deliverables:** Configured development environments, GitHub repo with board, dataset download confirmation, initial contact list for usability feedback.

---

### Phase 1: Data Pipeline & Preprocessing (Weeks 2–3)

**Objective:** Build a robust, reusable pipeline that converts raw video/frames into normalized landmark sequences for model training.

**Tasks:**

1. **Data cleaning & organization:**
    - Standardize all samples: label format, file naming, removal of corrupt or ambiguous videos.
    - Split dataset per sign into train (70%), validation (15%), test (15%) sets, stratified by sign.
2. **Landmark extraction with MediaPipe:**
    - Develop a script (`extract_landmarks.py`) that uses MediaPipe Holistic or Hands to process each video frame-by-frame, extracting 21 hand landmarks (x,y,z per point) and optionally pose landmarks.
    - Save each sequence as a `.npy` file: a 3D array `(T, 63)` for one hand or `(T, 126)` for both hands.
    - Implement temporal normalization: pad/truncate all sequences to a consistent length (e.g., 30 frames) using linear interpolation or repetition.
3. **Data augmentation (to improve robustness):**
    - Apply on-the-fly augmentations: slight rotations, scaling, translation, brightness/contrast adjustments, frame dropping.
    - Code these as TensorFlow/Keras layers or as part of the data generator to be fed into the LSTM.
4. **Pipeline validation:**
    - Visually verify a few samples by plotting landmark skeletons on frames.
    - Ensure no label leakage across splits.
5. **Create data loading utility:**
    - Implement a Keras `Sequence` or `tf.data.Dataset` generator that yields batches of (landmark_sequence, label).

**Deliverables:** Preprocessed training/validation/test sets (`.npy` format), landmark extraction script, data augmentation module, data generator class.

**In-charge:** Both members; one focuses on MediaPipe extraction optimisation, the other on dataset curation and augmentation logic.

---

### Phase 2: Model Development – LSTM (Weeks 4–6)

**Objective:** Train an LSTM-based network that achieves ≥85% accuracy on the validation set under controlled conditions.

**Tasks:**

1. **Model architecture design:**
    - Input: `(30, 126)` (both hands) or `(30, 63)` (dominant hand); start with both.
    - 2–3 LSTM layers (e.g., 128, 64 units) with dropout (0.3–0.5) and recurrent dropout.
    - Dense layers for classification with softmax; output size = number of recognized signs.
    - Consider adding batch normalization and residual connections if overfitting occurs.
2. **Training loop:**
    - Use categorical cross-entropy, Adam optimizer, early stopping, and ReduceLROnPlateau.
    - Train initially on static alphabet/digits, then fine-tune on dynamic WLASL subset.
    - Log experiments using TensorBoard: accuracy, loss, learning rate, confusion matrix per epoch.
3. **Hyperparameter tuning:**
    - Run systematic grid/random search (use a small subset) over: LSTM units, dropout rates, learning rate, batch size.
4. **Model evaluation:**
    - Compute accuracy, precision, recall, F1-score, confusion matrix on the test set.
    - Real-time feasibility test: measure inference time per sample (target <500 ms).
5. **Model export:**
    - Save final model as `.h5` and also convert to TensorFlow.js if client-side inference is desired later (optional).

**Deliverables:** Trained model file, evaluation report (metrics + confusion matrix), TensorBoard logs, inference time profiling.

**In-charge:** Primarily the member with stronger ML background; the other assists with data pipeline robustness and initial hyperparameter sweeps.

---

### Phase 3: Backend & API Development (Weeks 5–7)

**Objective:** Create a Flask/FastAPI server that hosts the model, handles real-time WebSocket streams, and exposes a REST API for translation.

**Tasks:**

1. **Server setup:**
    - Flask (or FastAPI) with Flask-SocketIO for real-time communication.
    - Organize code: `app.py`, `model_loader.py`, `socket_handlers.py`, `utils.py`.
2. **Model serving:**
    - Load the trained `.h5` model at startup.
    - Implement a function that receives a sequence of landmarks (accumulated from a moving window) and returns predicted class + confidence.
3. **WebSocket & REST endpoints:**
    - `socket`: receive landmark frames from frontend, maintain sliding window of latest N frames, run inference, emit predicted text.
    - `POST /speech-to-text`: wrapper around a browser-based Speech-to-Text API (Web Speech API on frontend) – backend stores conversation log.
    - `GET /transcript`: return current session’s text log.
4. **Conversation log storage:**
    - Simple SQLite database (or JSON file) recording timestamps, speaker (sign/text), and message.
5. **Performance optimization:**
    - Use threading or async workers so that multiple concurrent requests do not block.
    - Pre-load model once; use a lock for inference if needed.

**Deliverables:** Working backend with WebSocket endpoint, model inference integration, conversation transcript API.

**In-charge:** Member with backend/full-stack interest; collaborates with ML member to ensure correct model input shape and scaling.

---

### Phase 4: Frontend Development (Weeks 6–9)

**Objective:** Build a React-based UI that provides a seamless bidirectional communication experience.

**Tasks:**

1. **Project scaffolding:**
    - Create React app (`create-react-app` or Vite), install dependencies: Axios, socket.io-client, Web Speech API polyfills if needed.
2. **Webcam integration:**
    - Use the `getUserMedia` API to capture video; display it in a `<video>` element.
    - Render canvas overlay for MediaPipe drawing (if visual feedback desired).
3. **MediaPipe in-browser or video stream to backend:**
    - Two options: run MediaPipe Hands directly in the browser via MediaPipe’s JavaScript SDK (preferred for lower latency) and send landmarks via WebSocket, OR send raw frames to backend for MediaPipe (higher bandwidth). The proposal implies MediaPipe for spatial features; it is more efficient to run MediaPipe on the frontend and only transmit landmarks. This plan assumes **frontend MediaPipe**.
    - Implement a custom hook `useSignRecognition` that:
        - Initializes MediaPipe Hands.
        - On each frame, extracts landmarks, pushes to a buffer (rolling window), sends window over WebSocket.
        - Receives recognized text and displays it.
4. **Speech-to-text panel:**
    - Use Web Speech API (SpeechRecognition) to convert spoken English to text.
    - Display the transcribed text in the DHH user panel.
5. **UI layout:**
    - Dual-panel design: left panel for sign user (webcam + recognized text), right panel for hearing user (speech input + text log).
    - Conversation history log at the bottom or side.
    - Buttons: “Start/Stop Signing”, “Start/Stop Speaking”, “Export Transcript”.
6. **Accessibility & responsiveness:**
    - Follow basic WCAG guidelines: high contrast, large fonts, keyboard navigation.
    - Test responsiveness on standard laptop screens.

**Deliverables:** Fully functional React frontend, integrated with backend via WebSocket, webcam feed with hand landmark overlay, working speech-to-text panel.

**In-charge:** Member with frontend expertise; coordinates with backend developer on WebSocket message format.

---

### Phase 5: Full System Integration (Week 10)

**Objective:** Connect frontend, backend, and AI model into a single seamless pipeline; optimize for real-time performance.

**Tasks:**

1. **End-to-end connectivity:**
    - Deploy backend locally or on a cloud VM (e.g., AWS EC2 free tier) for testing; ensure frontend can connect.
    - Verify that landmark sequences flow from frontend MediaPipe → WebSocket → backend inference → recognized text → UI display.
    - Integrate speech-to-text so transcript updates in real time.
2. **Latency measurement & reduction:**
    - Measure end-to-end latency from gesture completion to text display.
    - Optimize: reduce frame window length if possible, use model quantization, or offload inference to a separate thread.
    - Target <2 seconds per response.
3. **Error handling & resilience:**
    - Handle webcam disconnect, server downtime, model not loaded states gracefully.
    - Display user-friendly error messages.
4. **Basic logging:**
    - Store conversation logs in SQLite and display them in UI on request.

**Deliverables:** Integrated prototype running on localhost with acceptable latency, error-handling mechanisms.

**In-charge:** Both members; pair-program critical integration points.

---

### Phase 6: Testing, Evaluation & Usability (Weeks 11–12)

**Objective:** Collect systematic performance metrics and gather formative user feedback to refine the prototype.

**Tasks:**

1. **Controlled environment testing:**
    - Use the test set created in Phase 0 to measure accuracy, precision, recall, F1-score again after integration (in case of drift).
    - Test with real-time video: 2 team members sign the full vocabulary under three lighting conditions (bright, dim, backlit) and two backgrounds (plain, cluttered).
    - Record all results: average accuracy drop, latency changes.
2. **Usability sessions:**
    - Conduct 2–3 structured sessions with DHH individuals or interpreters, as planned.
    - Prepare simple tasks: sign 10 common phrases, use speech-to-text to reply, view transcript.
    - Use a short feedback form (Likert scale + open questions) focusing on ease of use, clarity, and perceived usefulness.
3. **Bug tracking & refinement:**
    - Document all issues and prioritize critical ones (e.g., model crashes on fast signing, UI freezes).
    - Implement fixes in a sprint cycle.
4. **Performance benchmarking:**
    - Formal measure of FPS during real-time operation, model inference time, and memory usage.
    - Generate confusion matrix heatmap (Seaborn/Matplotlib) for the final model.

**Deliverables:** Test report (quantitative metrics), usability feedback summary with improvement notes, refined prototype.

**In-charge:** Both; one leads technical evaluation, the other leads usability sessions.

---

### Phase 7: Documentation & Defense Preparation (Weeks 13–14)

**Objective:** Prepare all final deliverables, technical report, and defense presentation.

**Tasks:**

1. **Technical documentation:**
    - System architecture diagram (updated).
    - API documentation (Swagger/OpenAPI for backend endpoints).
    - User manual: how to start the system, interpret the UI.
    - Setup guide (README): environment, dependencies, how to run.
2. **Project report writing:**
    - Draft final report chapters aligned with proposal structure: Introduction, Literature Review, Methodology, Implementation, Results & Discussion, Conclusion.
    - Include all evaluation charts, confusion matrix, usability findings.
    - Appendices: source code highlights, dataset sources, stakeholder consent forms.
3. **Demo video:**
    - Record a 3–5 minute demo showing the working system with real sign input and speech reply.
    - Add captions and narration.
4. **Pre-defense rehearsal:**
    - Prepare a 15-minute slide deck summarizing problem, solution, methodology, demo, results, and future work.
    - Practice with peers/supervisor; refine based on feedback.
5. **Finalize repository:**
    - Ensure code is clean, commented, with `requirements.txt` and `package.json`.
    - Tag a final release `v1.0-prototype`.

**Deliverables:** Final report, user guide, demo video, presentation slides, public GitHub repository.

**In-charge:** Both members divide writing and presentation tasks; one leads video and demo preparation.

---

## 3. Resource Allocation & Team Roles

Given a two-member team (Reg #222018025 and #222011091), responsibilities can be split but overlapping to ensure knowledge transfer:

| Role | Primary Responsibilities | Member A | Member B |
| --- | --- | --- | --- |
| ML/AI Lead | Data pipeline, model design, training, evaluation, MediaPipe extraction script | ✔ | (assists with dataset and testing) |
| Full-Stack & Backend Lead | Flask server, WebSocket, model serving, deployment | (assists with API design) | ✔ |
| Frontend Lead | React UI, MediaPipe in-browser, webcam, speech-to-text | (backup) | ✔ |
| QA & Usability Lead | Test cases, usability sessions, performance metrics, documentation | ✔ | ✔ |
| Project Manager | Timeline tracking, meetings, stakeholder coordination | Shared (weekly rotation) |  |

**Tools & Infrastructure:**

- GitHub with project board, issues, branches.
- Google Drive for shared documents, datasets (limited), and feedback forms.
- Local development machines; optionally a cloud GPU (Google Colab Pro or university lab) for initial training if needed.

**Weekly cadence:** Two stand-up meetings, one supervisor sync, and one dedicated deep-work session.

---

## 4. Risk Management

| Risk | Probability | Impact | Mitigation Strategy |
| --- | --- | --- | --- |
| Insufficient dynamic ASL video dataset | Medium | High | Use WLASL subset first; if not enough samples, employ frame-sequence augmentation or collect 5–10 signs via self-recording (with consent). Fallback: restrict to fingerspelling + numbers initially. |
| Model accuracy below 85% under real-world variations | Medium | High | Implement data augmentation aggressively; use transfer learning from a pre-trained hand gesture model; limit environmental conditions in scope. |
| Real-time latency >2s | High | Medium | Offload MediaPipe to frontend (JavaScript), reduce sequence window, use quantization (TensorFlow Lite), profile backend inference and optimize. |
| Poor usability feedback from DHH stakeholders | Low | Medium | Engage early and incorporate feedback iteratively; accept constructive criticism and adjust UI/UX. |
| Scope creep (wanting to add too many signs) | Medium | High | Strictly adhere to planned vocabulary; additional signs only if core MVP is fully working at Week 10. |
| Hardware/software compatibility issues | Medium | Low | Test on multiple browsers/machines during integration; use standard Web APIs with polyfills. |

---

## 5. Key Deliverables and Milestones

| Milestone | Week | Criteria |
| --- | --- | --- |
| Project Greenlight | 1 | Env ready, datasets downloaded, stakeholder contacts confirmed |
| Landmark Pipeline Complete | 3 | Preprocessing scripts output validated sequences |
| Model Trained & Evaluated | 6 | Validation accuracy ≥85%, inference time <500ms |
| Backend API Working | 7 | WebSocket serves predictions, speech-to-text endpoint functional |
| Frontend UI Standalone | 9 | Webcam + MediaPipe + speech-to-text working, connected to backend |
| Integrated Prototype | 10 | End-to-end conversation possible, latency acceptable |
| Formal Evaluation & Usability | 12 | Test reports, usability summary, bugs fixed |
| Final Submission Package | 14 | Final report, video, slides, code repository tagged |