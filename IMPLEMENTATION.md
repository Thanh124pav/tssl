# VASOT/VAOT reproduction, limitation diagnosis, and continuous-curve experiments

## 1. Mục tiêu và cách dùng tài liệu

Tài liệu này chia công việc thành các **session packet nhỏ** để Claude có thể hoàn thành từng phần mà không phải giữ toàn bộ thí nghiệm trong một session. Mỗi session chỉ nên làm **một packet**, cập nhật trạng thái trong file này, tạo deliverable có thể kiểm tra được, rồi commit.

Ba mục tiêu cuối cùng:

1. Đóng reproduction của VASOT, đặc biệt Desktop Assembly (DA) và YouTube Instructions (YTI).
2. Reproduce VAOT như một track độc lập, với code, data, environment và metrics riêng.
3. Kiểm chứng ba limitation hypothesis A/B/C và đánh giá proposal làm trơn discrete representations bằng arc-length, Gaussian, B-spline và Bézier.

DA và YTI thuộc track **VASOT action segmentation**. Không được dùng chúng như bằng chứng rằng VAOT đã được reproduce. VAOT dùng protocol/dataset/metrics khác.

## 2. Trạng thái hiện tại — đọc trước mỗi session

### 2.1 Repository và runtime

- Working tree chính: `/mnt/data/thanhpv/tssl`
- Branch: `main`
- Python environment: `acvcl`
- VASOT source: `reproductions/tssl/repos/VASOT`
- Data link hiện tại:
  `reproductions/tssl/repos/VASOT/data -> /mnt/data/thanhpv/acvcl/reproductions/tssl/data/final`
- Artifact của các session cũ vẫn chủ yếu nằm dưới:
  `/mnt/data/thanhpv/acvcl/reproductions/tssl/`
- VAOT source hiện mới tồn tại dưới dạng nested repository cũ:
  `/mnt/data/thanhpv/acvcl/reproductions/tssl/repos/VAOT`

Không xóa workspace cũ cho đến khi packet S00 hoàn tất và artifact đã được kiểm kê.

### 2.2 VASOT reproduction đã chạy

Các run sau đã kết thúc với exit code 0. Đây là kết quả **provisional**, chưa phải reproduction đã đóng vì còn phải audit protocol, aggregate YTI và nhận diện artifact/checkpoint.

| Dataset/activity | Seed | MoF | F1 | mIoU |
|---|---:|---:|---:|---:|
| Desktop Assembly | 0 | 0.598555 | 0.637993 | 0.421149 |
| YTI changing tire | 0 | 0.448281 | 0.295638 | 0.203974 |
| YTI coffee | 0 | 0.684720 | 0.471985 | 0.324918 |
| YTI CPR | 0 | 0.558960 | 0.492422 | 0.272710 |
| YTI jump car | 0 | 0.339239 | 0.260429 | 0.174356 |
| YTI repot | 0 | 0.365547 | 0.226080 | 0.153514 |

Nguồn hiện tại:

- Status: `/mnt/data/thanhpv/acvcl/reproductions/tssl/logs/session_full/_status.tsv`
- Logs: `/mnt/data/thanhpv/acvcl/reproductions/tssl/logs/session_full/repro_*.log`
- Dispatcher log: `/mnt/data/thanhpv/acvcl/reproductions/tssl/logs/session_full/_dispatcher.log`

Các vấn đề chưa đóng:

- Public runner/README dùng DA 30 epochs, `rho=0.25`; paper protocol có dấu hiệu dùng 100 epochs, `rho=0.2`. S01 phải kiểm tra nguồn và đóng băng protocol trước khi quyết định có rerun hay không.
- YTI phải được aggregate theo đúng quy tắc báo cáo của paper trên cả 5 activities.
- Nhiều process dùng chung `lightning_logs/version_62534`; checkpoint có suffix va chạm và không thể map chắc chắn về từng run.
- `train.py` đang khởi tạo `Trainer` hai lần khi có `VASOT_CKPT_DIR`.
- Không được rerun DA/YTI chỉ vì status cũ trong `REPRODUCTION_STATUS.md` nói “blocked”; status đó đã stale.

### 2.3 Data đã được kiểm tra

- Desktop Assembly: 129 video feature/GT match, feature 512 chiều, gồm 76 `fixed_order` và 53 `permuted_missing`; loader đọc được toàn bộ.
- 34/129 video DA có feature dài hơn GT đúng một frame. Loader hiện index theo GT nên bỏ trailing feature. Phải ghi rõ quyết định protocol này trong S01.
- YTI: 149 cặp feature/GT, feature 3000 chiều; loại file `_idt`; `repot_0026` thiếu ở cả feature và GT và đã được filter nhất quán.
- CPU smoke test đã qua cho DA và YTI, gồm forward, backward, optimizer update và Lightning train/val one-batch.

### 2.4 Continuous-curve work đã có

- Đã có `arc_reparam.py`, `bspline.py`, `gauss_smooth.py`, integration trong `train.py`, và `tests/test_continuous_curve.py`.
- Baseline và E1 arc-length trên FSeval, seed 0/1, đã kết thúc.
- Bốn run E2 B-spline FSeval bị ngắt khoảng epoch 9; chưa có test metrics hợp lệ.
- Checkpoint E2 có thể tồn tại dưới `lightning_logs/version_62534/checkpoints`, nhưng mapping run không đáng tin. Chỉ resume nếu S00/S02 xác định được identity bằng config/hparams; nếu không, rerun sau khi runner đã được sửa.

Kết quả chẩn đoán sơ bộ hiện có trong `reports/BSPLINE_RETROFIT_PLAN.md`:

- Normalized row entropy của transport: FS 0.077, FSeval 0.055 — rất peaky.
- Boundary z-score: FS 0.026, FSeval 0.047 — tín hiệu boundary yếu.
- Corr(index distance, semantic difference): FS 0.375, FSeval 0.384.

Đây mới là evidence sơ bộ, chưa đủ để kết luận nhân quả.

## 3. Quy tắc vận hành cho mọi session

Mỗi session Claude phải:

1. Chạy `git status --short --branch` và đọc phần “Trạng thái hiện tại” cùng packet được giao.
2. Không tự khởi chạy `run_full_session.sh` hoặc `overnight_sweep.sh`. Hai script này là runner lịch sử, không phù hợp với interactive session.
3. Chỉ làm một packet; tối đa 1–4 training run trong một packet, trừ packet chỉ launch một batch job đã có manifest.
4. Mọi run phải có `run_id` duy nhất, config snapshot, git commit, environment, seed, data fingerprint, log, checkpoint directory và exit status.
5. Không overwrite log/checkpoint/manifest cũ. Resume phải ghi `parent_run_id` và checkpoint nguồn.
6. Trước GPU run, kiểm tra GPU visibility, free memory và process đang chạy. CPU session chỉ làm audit, test, instrumentation hoặc tiny smoke test.
7. Job dài phải survive việc Claude hết session bằng scheduler hoặc `nohup`, ghi PID/status. Không ngồi chờ toàn bộ training trong cùng session.
8. Kết thúc packet bằng test/validation tương ứng, cập nhật ledger trong file này và commit code/docs. Không commit dataset, secrets, raw logs hay checkpoint.
9. Nếu packet phát hiện assumption sai, dừng trước khi mở rộng grid; ghi evidence và sửa dependency của các packet sau.

Mẫu handoff cuối session:

```text
Packet: Sxx
Commit: <hash hoặc "not committed">
Done: <deliverables>
Validation: <commands + result>
Runs launched/completed: <run_id + status>
Artifacts: <paths>
Blockers: <none hoặc chi tiết>
Next packet: <ID và command đầu tiên>
```

## 4. Artifact và runner contract cần đạt

S01/S02 nên chuẩn hóa layout sau; tên cụ thể có thể điều chỉnh nhưng contract không được mất:

```text
reproductions/tssl/
├── configs/
│   ├── vasot/baselines/
│   ├── vasot/diagnostics/
│   ├── vasot/curves/
│   └── vaot/
├── scripts/
│   ├── preflight.py
│   ├── run_one.py
│   └── collect_runs.py
├── artifacts/                 # gitignored runtime output
│   ├── run_registry.jsonl
│   └── runs/<run_id>/
│       ├── manifest.json
│       ├── config.yaml
│       ├── stdout.log
│       ├── status.json
│       ├── metrics.json
│       └── checkpoints/
└── reports/
```

`manifest.json` tối thiểu phải có:

- `run_id`, `track`, `dataset`, `activity`, `variant`, `seed`
- absolute data path và data fingerprint
- git commit, CLI/config đầy đủ, Python/CUDA/package snapshot
- start/end time, hostname, device, parent/resume information
- output paths và final exit state

`collect_runs.py` chỉ đọc manifest + metrics; không parse tên checkpoint để đoán run identity.

## 5. Session packets

### Phase S — Stabilize state và reproduction infrastructure

#### [x] S00 — Inventory và migrate artifact cũ

**Mục tiêu:** biến kết quả hiện có thành inventory đọc được từ repo mới mà không copy dataset/checkpoint ồ ạt.

**Việc làm:**

- Inventory status, logs, diagnostics và checkpoint trong workspace cũ; ghi size, checksum và nguồn.
- Tạo manifest hồi cứu cho 6 run DA/YTI đã hoàn tất và 4 E2 run bị ngắt.
- Dùng symlink có chủ đích hoặc runtime-path config cho artifact lớn; không đưa artifact vào Git.
- Map checkpoint về run chỉ khi hparams/checkpoint metadata đủ chứng minh. Nếu ambiguous, đánh dấu `checkpoint_identity=unknown`.
- Cập nhật `REPRODUCTION_STATUS.md` để bỏ trạng thái DA/YTI “blocked” đã stale.

**Deliverable:** artifact inventory + retrospective manifests + báo cáo checkpoint mapping.

**Exit criterion:** cả 6 baseline run có log và metrics truy được từ repo mới; ambiguity được ghi rõ, không đoán.

#### [x] S01 — Freeze protocol và config registry

**Mục tiêu:** xác định chính xác “reproduce” nghĩa là chạy config nào và báo cáo metrics nào.

**Việc làm:**

- Đối chiếu paper, official README/scripts và code default cho DA/YTI.
- Ghi riêng `paper_config`, `public_repo_config`, và `chosen_reproduction_config` nếu chúng khác nhau.
- Chốt DA 30/100 epochs và `rho=0.25/0.2` bằng citation tới nguồn trong repo hoặc paper.
- Chốt YTI activity list, K, missing-video policy và macro/micro aggregation.
- Chốt policy DA trailing feature frame.
- Tạo machine-readable configs; không để hyperparameter chỉ tồn tại trong shell script.

**Deliverable:** `PROTOCOL.md` hoặc section tương đương, baseline configs và bảng discrepancy.

**Exit criterion:** có thể tạo exact CLI từ config; biết rõ run hiện tại có khớp protocol đã chọn hay cần rerun.

#### [x] S02 — Refactor single-run runner và checkpoint isolation

**Mục tiêu:** một run tạo đúng một Trainer và một artifact directory duy nhất.

**Việc làm:**

- Xóa logic khởi tạo `Trainer` hai lần trong `train.py`.
- Thêm accelerator/device auto-detection để cùng code chạy CPU smoke và GPU training.
- Thêm explicit run directory, unique logger version, checkpoint filename và resume path.
- Tạo `run_one.py`/wrapper có manifest, atomic status update và non-zero failure state.
- Thêm test cho run-id collision, completed-run skip và resume provenance.

**Deliverable:** isolated runner + unit/integration tests.

**Exit criterion:** hai tiny runs liên tiếp không chia log/checkpoint; một failed run không được đánh dấu completed.

#### [x] S03 — End-to-end preflight trên CPU và GPU

**Mục tiêu:** chứng minh data → model → loss → backward → validation → checkpoint → resume → test thông luồng.

**Việc làm:**

- CPU: tiny DA và một YTI activity, 1–2 batch, giới hạn frame/video.
- GPU: chỉ chạy khi GPU có sẵn; một tiny run và resume thêm một epoch.
- Kiểm tra metrics JSON, manifest và checkpoint load.

**Exit criterion:** cả hai dataset đi hết pipeline; nếu session không có GPU thì CPU gate đủ để tiếp tục audit, nhưng chưa launch training dài.

### Phase R — Close VASOT DA/YTI reproduction

#### [x] R00 — Audit Desktop Assembly baseline

**Mục tiêu:** quyết định run DA hiện tại có thể dùng hay phải chạy lại.

**Việc làm:**

- Kiểm tra config/log/data fingerprint của result seed 0.
- So sánh đúng paper/public protocol đã freeze tại S01.
- Recompute metrics từ saved predictions nếu có.
- Nếu protocol khớp: đóng run hiện tại. Nếu không khớp: launch đúng **một** canonical run; không mở multi-seed grid.
- Phân tích riêng 34 video lệch một trailing frame để chắc metric không bị silent mismatch.

**Exit criterion:** một DA canonical run có provenance đầy đủ và bảng so sánh paper.

#### [x] R01 — Audit và aggregate YTI baseline

**Mục tiêu:** đóng kết quả YTI trên cả 5 activities.

**Việc làm:**

- Validate từng run/status/log và exact video set.
- Tính aggregate theo protocol của paper; nếu paper dùng mean activity thì không thay bằng pooled-frame metric.
- Báo cáo per-activity lẫn aggregate, chênh lệch với paper và uncertainty chưa biết do chỉ seed 0.
- Chỉ rerun activity bị lỗi/protocol mismatch.

**Exit criterion:** một bảng YTI canonical với công thức aggregation và provenance.

#### [ ] R02 — Reproduction variance, chỉ khi cần

**Mục tiêu:** định lượng variance thay vì mở rộng seed không có lý do.

**Gate:** chỉ chạy nếu paper comparison không khớp đáng kể, hoặc proposal sau này cần paired seeds.

**Việc làm:** seed 1 cho DA và/hoặc 5 YTI activities; seed 2 chỉ cho final claim hoặc variance quá lớn.

**Exit criterion:** mean/std theo cùng seed set; quyết định rõ mismatch là protocol, stochasticity hay implementation.

### Phase D — Diagnose limitation hypotheses

#### [x] D00 — Diagnostic schema và golden tests

**Mục tiêu:** instrumentation không làm thay đổi training semantics.

**Việc làm:**

- Định nghĩa snapshot tại init, đầu/mid/cuối training và test.
- Tách `T_xa` segmentation transport khỏi `T_xy` alignment transport.
- Lưu summary statistics mặc định; raw matrix chỉ cho sample nhỏ để tránh disk explosion.
- Golden test: bật diagnostics không làm đổi loss/metrics trong tolerance.

**Exit criterion:** một tiny run xuất schema versioned và reproduce cùng loss khi diagnostics off/on.

#### [x] D01 — Hypothesis A: T* peaky và supervision nghèo

**Hypothesis:** FGW với entropy và diagonal priors tạo transport peaky; stop-gradient CE gần top-1 positive supervision, ít negative/repulsive signal.

**Đo:**

- normalized row entropy, effective support `exp(H)`, top-1/top-k mass và Gini/sparsity;
- diagonal distance/width của mass;
- gradient density, norm và số pair nhận signal hữu dụng;
- init so với trained; `T_xa` so với `T_xy`; per-dataset/per-action duration.

**Controls:** thay epsilon/temperature hoặc tắt từng prior trên một dataset nhỏ. Không thay nhiều yếu tố trong cùng run.

**Exit criterion:** kết luận support/refute/conditional với effect size, không chỉ heatmap đẹp.

#### [x] D02 — Hypothesis B: tautological intra-video loss và boundary blindness

**Hypothesis:** target segmentation được sinh từ cùng cost đang tối ưu; model học clustering frame-level nhưng không học transition sharpness.

**Đo:**

- boundary precision/recall/F1 với nhiều tolerance;
- boundary displacement, fragmentation và over-segmentation ratio;
- change-point AUROC/AUPRC từ logit difference, embedding difference và curvature;
- so sánh interior frames với vùng gần GT boundary;
- label permutation/control để xem pseudo-target có tạo vòng lặp tự xác nhận hay không.

GT chỉ dùng cho diagnosis/evaluation, không được vô tình đi vào training.

**Exit criterion:** chỉ ra loss giảm có hay không tương quan với boundary quality và failure mode cụ thể.

#### [x] D03 — Hypothesis C: frame index là prior bất lợi

**Hypothesis:** khoảng cách index sau uniform sampling không đại diện semantic/time distance, nhất là action duration không đều.

**Đo:**

- correlation giữa index distance và feature/semantic distance;
- warping/off-diagonal mass theo action duration;
- error theo short/long action và gần boundary;
- ablation giảm/tắt `R` và `C^x`, so với arc-length reparameterization;
- dùng original timestamps/source-frame indices nếu dataset còn metadata. Nếu không có, ghi rõ giới hạn, không giả lập thành ground truth.

**Exit criterion:** effect theo duration/boundary và ít nhất một controlled prior ablation.

#### [x] D04 — Aggregate diagnosis report

**Mục tiêu:** tổng hợp A/B/C trước khi gán cơ chế cho smoothing.

**Deliverable:** report gồm hypothesis, metric, dataset, seed, control, effect size, counter-evidence và confidence.

**Exit criterion:** mỗi hypothesis được gắn `supported`, `partially supported`, `not supported`, hoặc `inconclusive`, cùng lý do.

### Phase V — VAOT reproduction độc lập

#### [x] V00 — Import VAOT source và pin environment

**Mục tiêu:** đưa VAOT vào repo mà không mang nested `.git` và không làm hỏng VASOT env.

**Việc làm:**

- Record upstream URL và commit của `VAOT-ICCV25`.
- Vendor/subtree source vào `reproductions/tssl/repos/VAOT`; không commit `.git` lồng nhau.
- Inventory dependency, PyTorch Lightning API cũ và CUDA compatibility.
- Ưu tiên environment/lock riêng cho VAOT; chỉ patch compatibility khi có smoke test bảo vệ.

**Exit criterion:** clean import, provenance rõ, import test chạy được.

#### [ ] V01 — Audit VAOT data và evaluation protocol

**Mục tiêu:** xác định chính xác những gì đang có và còn thiếu.

**Việc làm:**

- Inventory Pouring, Penn Action, IKEA ASM raw frames, labels và `npyrecords`.
- Trace code tạo/đọc progress labels; validate split và sample counts.
- Đóng băng metrics VAOT riêng: Acc tại tolerance, progress, Kendall Tau, AP@5/10/15 theo paper/code.
- Không coi feature của DA/YTI là input thay thế hợp lệ.

**Exit criterion:** data matrix `available/missing/invalid` và một protocol config cho dataset đầu tiên.

#### [ ] V02 — VAOT one-batch smoke

**Mục tiêu:** thông pipeline trên dataset nhỏ nhất sẵn có, ưu tiên Pouring nếu data đầy đủ.

**Exit criterion:** data load, forward, loss, backward, validation/evaluation và checkpoint đều qua; không cần metric tốt.

#### [ ] V03 — VAOT canonical baseline đầu tiên

**Mục tiêu:** reproduce một dataset end-to-end với provenance đầy đủ.

**Exit criterion:** table paper-vs-local và deterministic evaluation từ checkpoint.

#### [ ] V04 — VAOT remaining datasets và final aggregate

**Gate:** V03 phải ổn định; data/license phải có sẵn.

**Exit criterion:** chạy từng dataset như packet riêng nếu dài; final report không trộn metrics VAOT với VASOT.

### Phase E — Continuous representations proposal

#### [ ] E00 — Freeze experimental contract

**Mục tiêu:** tránh sửa baseline trong lúc so sánh proposal.

**Việc làm:**

- Chọn development ladder: FSeval → FS Mid → DA/YTI; không scale ngay tất cả dataset.
- Paired seeds: 0/1 cho screening, seed 2 cho winner/final claim.
- Freeze baseline commit/config/data split/metrics.
- Định nghĩa compute budget và stopping rule.

**Scale gate đề xuất:** chỉ scale variant nếu cải thiện ít nhất 1 percentage point trung bình qua 2 seeds, hoặc cải thiện diagnostic quan trọng mà task metric giảm không quá 1 point. Báo cả effect từng seed; không cherry-pick.

#### [ ] E01 — Recover hoặc rerun minimal B-spline E2

**Mục tiêu:** hoàn tất một B-spline run đáng tin trên FSeval.

**Việc làm:**

- Sau S02, xác định E2 checkpoint cũ có resume an toàn không.
- Nếu identity không chắc, bỏ resume và chạy lại một seed với isolated runner.
- Verify interpolation shape, endpoint behavior, gradients, short-sequence fallback và no-NaN.

**Exit criterion:** một E2 seed hoàn tất với manifest + metrics + diagnostic A/B/C.

#### [ ] E02 — B-spline controlled two-seed comparison

**Mục tiêu:** so baseline, arc-length và B-spline với cùng seed/config.

**Ablation tối thiểu:** representation space (`X` hoặc logits), blend nhỏ trước; không đồng thời đổi degree, knots, loss và optimizer.

**Exit criterion:** paired two-seed table và quyết định stop/scale.

#### [ ] E03 — Gaussian control

**Mục tiêu:** phân biệt lợi ích của continuous curve với generic low-pass smoothing.

**Việc làm:** một hoặc hai sigma đã pre-register, cùng blend/warmup budget như B-spline.

**Exit criterion:** biết B-spline có hơn Gaussian control hay chỉ là smoothing effect.

#### [x] E04 — Piecewise cubic Bézier implementation

**Mục tiêu:** thêm Bézier control có tính local và testable.

**Lưu ý thiết kế:** dùng piecewise cubic Bézier/Bézier spline với endpoint interpolation và continuity rõ ràng; tránh một global high-degree Bézier curve dễ oversmooth và bất ổn số.

**Tests:** shape, endpoints, continuity, gradient, short sequence, identity/zero blend, CPU/GPU consistency trong tolerance.

**Exit criterion:** unit tests + one-batch smoke; chưa training grid.

#### [ ] E05 — Small controlled model selection

**Mục tiêu:** so baseline, arc, Gaussian, B-spline và Bézier trên cùng FSeval matrix.

**Giới hạn:** tối đa 2 seeds/variant; successive halving nếu variant rõ ràng kém hoặc bất ổn.

**Đo:** MoF/F1/mIoU, boundary metrics, T entropy/support, curvature, runtime, memory và stability.

**Exit criterion:** chọn tối đa 1–2 winner và ghi cả negative result.

#### [ ] E06 — Scale winner sang FS Mid, DA và YTI

**Gate:** E05 qua scale gate; VASOT baseline R00/R01 đã đóng.

**Việc làm:** chạy từng dataset/activity thành packet riêng nếu dài. Dùng paired seed và exact baseline config.

**Exit criterion:** cross-dataset evidence; không claim generality chỉ từ FSeval.

#### [ ] E07 — Mechanism controls cho limitation A/B/C

**Mục tiêu:** tránh kết luận “curve giải quyết limitation” chỉ vì metric tăng.

**Controls khả thi:** epsilon/temperature, prior strength, top-k/soft target, explicit repulsion/contrastive term, boundary regularizer. Mỗi lần chỉ thay một cơ chế.

**Exit criterion:** curve effect được so với direct intervention lên hypothesis tương ứng.

#### [ ] E13 — Action bridge giữa segmentation và alignment

**Mục tiêu:** biến action space thành cầu nối hai chiều, thay vì hai task chỉ
chia sẻ encoder. Action posterior của mỗi frame bổ sung một cost cho
frame-to-frame alignment; transport alignment đồng thời giám sát action
posterior qua một auxiliary loss nhỏ.

**Ablation bắt buộc:** action-cost only và bidirectional action bridge. Không
dùng spline trong packet này để tách gain của action bridge khỏi smoothing.

**Đo:** MoF/F1/mIoU, boundary F1, transport support, action-consistency của
alignment. Không gọi proxy action-consistency là paper alignment metric.

**Exit criterion:** paired seeds 0/1 trên FSeval, checkpoint finite, effect
từng seed và quyết định stop/scale theo E00.

#### [ ] E14 — B-spline fit-quality và hyperparameter sensitivity

**Mục tiêu:** phân biệt “curve fit không đạt” với “curve fit tốt nhưng
representation/task metric kém”.

**Diagnostics bắt buộc:** reconstruction RMSE/max error theo frame, số control
points/knots, tỷ lệ chạm `k_max`, knot density quanh GT/predicted boundaries,
curvature, boundary-vs-interior error và runtime. Sweep có kiểm soát `tol`,
`k_max`, degree và blend; mỗi lần chỉ đổi một trục.

**Quy tắc kết luận:** chỉ gọi một cấu hình B-spline là downstream failure nếu
fit diagnostics đạt ngưỡng pre-register và checkpoint finite. Fit failure,
numerical failure và downstream failure phải ghi riêng.

**Exit criterion:** sensitivity surface nhỏ trên checkpoint/batches cố định,
sau đó tối đa hai cấu hình hợp lệ được train paired seeds.

#### [ ] E15 — Boundary-aware action-space B-spline

**Mục tiêu:** smooth action trajectory trong action interior nhưng giữ nguyên
hoặc giảm blend ở transition. Boundary gate phải self-supervised ở train;
GT chỉ dùng diagnostic/evaluation.

**Controls:** global action-space spline, boundary gate không spline, và
boundary-aware spline. So với E13 action bridge không spline.

**Exit criterion:** paired FSeval table gồm task metrics, direct boundary F1,
boundary count ratio và fit-quality diagnostics E14.

#### [ ] E16 — Segment-wise temporal scaling augmentation

**Mục tiêu:** sau khi có soft segmentation/alignment đủ ổn định, co giãn thời
gian từng segment/soft segment và buộc frame representation bất biến với tốc
độ thực hiện, trong khi bảo toàn thứ tự và boundary.

**Thiết kế tối thiểu:** sample monotone piecewise warp quanh các segment,
re-sample representation bằng curve, consistency giữa original/warped action
posterior và alignment transport. Không dùng GT boundaries khi train.

**Controls:** global speed warp, segment-wise hard warp, segment-wise soft
warp; report warp range và tỷ lệ frame đi qua boundary.

**Gate:** chỉ chạy sau E13–E15 hoặc khi pseudo-boundary reliability đạt ngưỡng
pre-register; nếu boundary nhiễu, temporal scaling sẽ khuếch đại label noise.

**Exit criterion:** paired task/boundary/alignment-proxy metrics và kiểm tra
không collapse dưới warp mạnh.


### Phase N — New theory: relational, transition, and reparameterization-aware alignment

**Research contract (2026-10-30).** Phase N is a NEW exploration track, not a reinterpretation of the completed limitation hypotheses D01–D04. Each direction has a low-risk **incremental/diagnostic** task and a gated **theory/novelty** task. Run the incremental task first; a better task score alone is not evidence of theoretical novelty. Preserve the frozen VASOT baseline and paired seeds; never use GT labels/boundaries to train. Distinguish the segmentation plan `T_xa` from alignment plan `T_xy` in every diagnostic. E09 found `T_xy` soft (not near one-hot); E14/E15 found poor low-capacity spline fit, which does not establish impossibility of a regular latent manifold. Do not claim causal identification from predictive influence or attention weights. A new method must beat appropriate simple controls, document prior art including non-CV work, and establish either a new identifiability/recovery property, a counterexample to the baseline, or a clearly justified optimization principle.

#### [ ] N-A1 — Incremental diagnostic: dynamics-aware structural cost

**Question:** Is frame-index proximity an inadequate proxy for within-video structural similarity, and can local dynamics features improve matching without changing GW itself?

**Tasks:**
- On frozen checkpoints quantify within-action/inter-action roughness, temporal neighbor rank, transition similarity and GT evaluation-only alignment errors, stratified by action duration and boundary distance.
- Compare the existing temporal-index `C^X,C^Y` with (i) feature-distance structure, (ii) local-window velocity/transition structure, and (iii) a learned dynamics encoder; report cost scale and sensitivity to feature jitter.
- Plug each structure into the SAME single-FGW solver, with the original cross-video cost, priors, marginals, budget and paired seeds unchanged. Use random/shuffled dynamics, index cost and no-structure controls. Record solver runtime and transport support.
- Explicitly distinguish an improvement from merely removing a harmful segmentation prior (`rho_seg`) or changing optimization difficulty.

**Deliverable:** `reports/NA1_DYNAMICS_COST_DIAGNOSIS.md`, comparable metrics, run manifests and at least one falsifiable failure case.

**Exit criterion:** a paired-seed table and mechanistic evidence showing whether dynamics-aware cost captures a relationship missed by temporal index; otherwise stop A2 scaling.

#### [ ] N-A2 — Theory/novelty: relational distributional transport

**Hypothesis:** A scalar temporal distance may not identify differences between stochastic transitions that are distinguishable from relational transition distributions.

**Tasks:**
- Define per-pair relational laws `Q^X_{ik}`, `Q^Y_{jl}` (conditional transition distributions, not arbitrary embeddings), plus a well-defined divergence `D(Q^X_{ik},Q^Y_{jl})`. State assumptions, sample availability, and what is identifiable from passive video.
- Formulate distributional relational transport with objective `sum_{ikjl} D(Q^X_{ik},Q^Y_{jl}) T_ij T_kl`; specify constraints, computational strategy and reduction to an existing scalar-GW baseline as a special case.
- Construct a controlled counterexample where frame-index or scalar-distance GW confuses distinct transition processes, with proof or explicitly scoped proposition and counterexample to overbroad claims.
- Compare against learned scalar GW, dynamics-cost N-A1, a generic distributional metric control, and existing relational/distributional OT literature (not only CV).

**Gate:** only proceed after N-A1 identifies a reproducible structural mismatch that distributions could plausibly distinguish.

**Exit criterion:** written mathematical statement with assumptions/proof or documented counterexample, complexity analysis and a synthetic test validating the distinguishing property before dataset-scale runs.

#### [ ] N-B1 — Incremental diagnostic: transition-aware matching and gradient conflict

**Question:** Does segmentation share action identity but fail to preserve within-action transition phase, explaining why E13 action bridge barely helps alignment?

**Tasks:**
- Measure same-action versus same-phase correspondence (phase labels for evaluation only where valid), within-action transition separability, transport stability under small embedding perturbations and gradient cosine of segmentation/alignment objectives.
- Implement a controlled local transition encoder and transition-cost-only single-OT baseline; compare state-only, action-posterior bridge E13, transition-cost-only and combined-cost under equal compute.
- Test temporal-window and directed-vs-symmetric transition features; record whether gains are from phase correspondence or merely higher same-action transport mass.
- Do not equate directional temporal prediction with intervention-defined causality.

**Deliverable:** `reports/NB1_TRANSITION_ALIGNMENT_DIAGNOSIS.md`, phase and alignment-proxy metrics with paired seeds.

**Exit criterion:** isolate at least one repeatable mismatch between state/action matching and transition correspondence, or document negative evidence and stop B2 escalation.

#### [ ] N-B2 — Theory/novelty: transition-commutative transport

**Hypothesis:** A correct correspondence should intertwine state-transition operators, not merely preserve pairwise temporal distances.

**Tasks:**
- Model video dynamics as stochastic kernels `P_X,P_Y` and normalize the alignment `A` consistently; study the commutator residual `||P_X A - A P_Y||` with compatible matrix dimensions and nontrivial marginal/anti-collapse constraints.
- Derive when zero/small commutator implies useful correspondence, including counterexamples from repeated states, stationary processes, lumpability and operator symmetries. Clearly distinguish predictive transitions from causal mechanisms.
- Develop an optimization method coupling (F)GW with commutativity; compare to direct transition-cost OT and an attention-only transition predictor. Analyze computational cost and the effect of pseudo-transition noise.
- Test synthetic Markov systems with known correspondence and confounders before FSeval. Explore time-rescaled operators only AFTER the unwarped case works (N-C2).

**Gate:** N-B1 demonstrates information in transitions that existing action/state matching misses.

**Exit criterion:** formal proposition/counterexample plus nondegenerate recovery experiment and controlled paired-seed comparison, not just improved MoF.

#### [ ] N-C1 — Incremental diagnostic: nonlinear temporal-warp stress test

**Question:** Does VASOT's temporal-index FGW impose structural penalties on semantically corresponding videos after local speed changes?

**Tasks:**
- Construct monotone piecewise temporal warps of the SAME video and of aligned video pairs; preserve action ordering and report interpolation policy, speed ratios and warp severity.
- Evaluate the exact current `C^X,C^Y`, diagonal `rho_align` and transport solver separately. Measure oracle-correspondence structural distortion, alignment error, solver feasibility, and transport entropy/support at fixed checkpoint.
- Controls: unwarped, global speed warp, piecewise warp, fixed-index FGW, index-prior removal, standard DTW/soft-DTW or Gromov-DTW where applicable; disentangle cost effects from interpolation artifacts.
- B-spline is OPTIONAL to parametrize a *monotone scalar time warp*, NOT to fit high-dimensional frame embeddings (E14/E15); first use linear/piecewise interpolation.

**Deliverable:** `reports/NC1_TEMPORAL_WARP_STRESS.md` plus a reproducible counterexample and sensitivity plots.

**Exit criterion:** identify a reproducible warp regime where the correct correspondence incurs avoidable structural distortion and establish whether the effect is from GW structure or diagonal prior; otherwise stop C2 scaling.

#### [ ] N-C2 — Theory/novelty: quotient-space, time-reparameterized transport

**Hypothesis:** Trajectories representing the same ordered dynamics should be compared modulo admissible monotone time reparameterizations, potentially together with transition intertwining.

**Tasks:**
- Define admissible warp class `Gamma`, equivalence relation (if truly valid), objective on trajectory classes and regularity/non-collapse constraints. State when an infimum over warps yields actual invariance versus only regularized robustness.
- Formulate joint alignment-and-warp optimization; compare explicit warp functions (monotone splines with positive increments) with a standard Gromov-DTW/elastic alignment baseline. Prove or falsify invariance/recovery under clearly scoped assumptions.
- Analyze identifiability and degeneracies (repeated actions, constant paths, noninvertible warps, partial/missing steps). Establish whether dynamics-intertwining (N-B2) supplies strictly new constraints; do not claim it without a separating example.
- Validate on synthetic nonlinear speed changes before paired FSeval; record warp estimation, alignment correctness, computational cost and original action-segmentation metrics.

**Gate:** N-C1 establishes a structural warp failure not solved by a simple existing alignment baseline, or provides a theory-motivated case worth testing.

**Exit criterion:** an explicit mathematical claim and proof/counterexample, comparison with non-CV elastic/SRVF and Gromov-DTW prior work, and synthetic evidence that isolates a novel advantage.

**Recommended dependency:** `E09/E13/E14 → N-A1, N-B1, N-C1`; `N-A1 → N-A2`; `N-B1 → N-B2`; `N-C1 → N-C2`. Do not launch A2/B2/C2 together. Revisit shared B2+C2 only if their separate failure modes are verified.

### Phase F — Finalize

#### [ ] F00 — Reproducibility audit và final report

**Việc làm:**

- Kiểm tra mọi table cell truy được tới manifest/config/log/commit.
- Phân biệt completed, failed, interrupted và not attempted.
- Báo mean/std, effect size, compute budget, deviations và negative results.
- Tạo lệnh tối thiểu để reproduce canonical VASOT, VAOT và winner proposal.
- Cập nhật README/status; archive metadata và checksum, không commit artifact lớn.

**Exit criterion:** một người khác có thể chọn một run_id, dựng lại config và giải thích con số trong final table.

## 6. Thứ tự thực hiện khuyến nghị

Critical path:

```text
S00 → S01 → S02 → S03
             ├─→ R00 → R01 → optional R02
             ├─→ D00 → D01/D02/D03 → D04
             ├─→ V00 → V01 → V02 → V03 → V04
             └─→ E00 → E01 → E02/E03 → E04 → E05 → E06 → E07
All completed tracks → F00
```

Không cần chờ VAOT xong mới làm diagnostics VASOT, nhưng không chạy hai training grid nặng đồng thời. Nếu session chỉ có CPU, ưu tiên S00–S02, D00 instrumentation, V00–V01, E04 tests và report aggregation.

### Execution matrix cho việc chọn session

“Run cap” là số training run tối đa được mở trong packet, không tính unit test/tiny smoke.

| Packet | Dependency | Resource phù hợp | Run cap | Kết quả chính |
|---|---|---|---:|---|
| S00 | none | CPU/storage | 0 | inventory + retrospective manifests |
| S01 | S00 | CPU | 0 | frozen protocols/configs |
| S02 | S01 | CPU | 0 | isolated runner + tests |
| S03 | S02 | CPU; GPU optional | 2 tiny | end-to-end preflight |
| R00 | S01–S03 | GPU chỉ nếu rerun | 1 | canonical DA result |
| R01 | S01–S03 | CPU; GPU nếu run lỗi | 1 | canonical YTI aggregate |
| R02 | R00/R01 + gate | GPU | 1–4 | variance estimate |
| D00 | S02 | CPU | 1 tiny | diagnostic schema/tests |
| D01 | D00 | GPU preferred | 1–2 | evidence cho hypothesis A |
| D02 | D00 | CPU/GPU | 1–2 | evidence cho hypothesis B |
| D03 | D00 | CPU/GPU | 1–2 | evidence cho hypothesis C |
| D04 | D01–D03 | CPU | 0 | diagnosis report |
| V00 | none | CPU | 0 | pinned VAOT source/env |
| V01 | V00 | CPU/storage | 0 | data/protocol audit |
| V02 | V01 | CPU; GPU optional | 1 tiny | VAOT pipeline smoke |
| V03 | V02 | GPU | 1 | first canonical VAOT baseline |
| V04 | V03 | GPU | 1 per session | remaining VAOT datasets |
| E00 | R00/R01 | CPU | 0 | frozen experiment matrix |
| E01 | S02 + E00 | GPU | 1 | completed B-spline run |
| E02 | E01 | GPU | 1–2 per session | paired B-spline comparison |
| E03 | E00 | GPU | 1–2 per session | Gaussian control |
| E04 | S02 | CPU | 0 | Bézier implementation/tests |
| E05 | E02–E04 | GPU | 1–2 per session | controlled model selection |
| E06 | E05 + gate | GPU | 1 per session | cross-dataset validation |
| E07 | D04 + E05 | GPU | 1–2 per session | mechanism controls |
| N-A1 | E09 + E14 + E00 | CPU/GPU | 1–2 | dynamics-cost diagnosis |
| N-A2 | N-A1 gate | CPU; GPU optional | 0–2 | distributional relational transport theory |
| N-B1 | E09 + E13 + E00 | CPU/GPU | 1–2 | transition diagnostic |
| N-B2 | N-B1 gate | CPU; GPU optional | 0–2 | commutative-transport theory |
| N-C1 | E09 + E14 | CPU/GPU | 0–2 | nonlinear-warp stress test |
| N-C2 | N-C1 gate | CPU; GPU optional | 0–2 | quotient temporal transport theory |
| F00 | completed tracks | CPU | 0 | final audit/report |

## 7. Session ledger

Claude phải cập nhật bảng này sau mỗi packet; chi tiết run nằm trong registry, không nhét raw log vào Markdown.

| Packet | Status | Commit/run IDs | Ghi chú |
|---|---|---|---|
| S00 | DONE | See reports/S00_INVENTORY.md | 14 manifests, inventory complete, REPRODUCTION_STATUS updated |
| S01 | DONE | See configs/PROTOCOL.md | DA=README config (30ep, rho=0.25), YTI=run_yti.sh, configs frozen |
| S02 | DONE | train.py refactored | Single Trainer, run isolation, manifest, auto CPU/GPU, metrics.py fix |
| S03 | DONE | CPU+GPU smoke pass | DA+YTI GPU smoke 1ep OK, FSeval CPU smoke OK, manifest+ckpt isolated |
| R00 | DONE | R00_DA_baseline_s0, wandb ge8a3wu7 | MoF=0.5986 F1=0.6380 mIoU=0.4211, canonical with provenance |
| R01 | DONE | R01_YTI_*_s0, wandb tihe3mu9/6jaj64ev/5cg9rker/+ | Mean: MoF=0.4793 F1=0.3493 mIoU=0.2259. All 5 match old runs exactly |
| R02 | DONE | R02_DA_baseline_s1, R02_YTI_*_baseline_s1 (6 runs) | Seed-1 baseline controls for E08 paired comparison |
| D00 | DONE | diagnose.py + 13 dataset runs | Schema versioned, diagnostics don't alter training |
| D01 | DONE | 13 datasets + E01 comparison | A: SUPPORTED — T_ent 0.024-0.119, but softening via B-spline hurts |
| D02 | DONE | 13 datasets | B: PARTIAL — boundary_z mean 0.145, negative for Breakfast |
| D03 | DONE | 13 datasets | C: SUPPORTED — idx_sem_corr 0.063-0.516, weakest on YTI |
| D04 | DONE | reports/D04_DIAGNOSIS_AGGREGATE.md | Full aggregate report with hypothesis verdicts |
| V00 | DONE | VAOT/PROVENANCE.md | Clean import, provenance, import test. PL 2.x porting needed (train.py, evaluations.py). Pouring data found but flat structure. Penn/IKEA/npyrecords missing. |
| V01 | DONE | reports/V01_VAOT_DATA_AUDIT.md | Pouring usable after reorg (.mov → per-video frames, labels from pkl); Penn/IKEA missing — V04 blocked |
| V03 | BLOCKED | | Pouring labels degenerate; no labelled VAOT dataset on cluster |
| V04 | BLOCKED | | Penn Action and IKEA ASM not present on cluster |
| E00 | DONE | reports/E00_EXPERIMENT_CONTRACT.md + E00_contract.lock | Contract backfilled, baseline config hash pinned |
| E01 | DONE | E01_bspline_FSeval_s{0,1} | B-spline paired seeds complete; mean MoF -11.95pp vs baseline |
| E02 | DONE | subsumed by E01 (paired-seed B-spline) | |
| E03 | DONE | E03_gauss_s2_FSeval_s{0,1} | Gaussian σ=2 paired seeds; mean MoF -10.55pp |
| E04 | RETIRED | bezier.py + 5 tests + train.py integration; E04_bezier_FSeval_s{0,1} | Checkpoints NaN (13 960/13 960 non-finite params) — numerical failure, not a legitimate result. Bézier dropped from roadmap per user. |
| E05 | DONE | reports/E05_COMPARISON_MATRIX.md | All variants fail scale gate; E06 skipped |
| E06 | SKIP | | Gate not passed on FSeval; scaling to DA/YTI unjustified |
| E07a | DONE (reinterpreted) | reports/E07a_EPS_ABLATION.md | ε↑ harms MoF monotonically; E09 shows this is partly a solver degeneration (T_xy collapses at ε≥0.10). Original "A refuted" claim softened. |
| E07b | DONE | reports/E07bc_PRIOR_ABLATION.md | rho ablation — rho=0 +7.95 pp FSeval; see E08 for cross-dataset verdict |
| E07c | DONE | reports/E07bc_PRIOR_ABLATION.md | radius_gw ablation — 0.12 FAIL (−2.85 pp), 0.50 FAIL (−15.20 pp) |
| E08 | DONE | reports/E08_RHO0_TRANSFER.md, E08a/b runs (12) + R02 (6) | rho=0 does NOT transfer: DA Δ=−3.23 pp, YTI mean Δ=−4.17 pp. FSeval gain is dataset-specific. |
| E09 | DONE | reports/E09_LIMITATION_RECHECK.md, scripts/eval_{boundary,rho_swap,transport_support}.py | Transport-support probes: T_xy is NOT peaky (support 30.87/256). ε control shows T_xy collapses with higher ε. Boundary F1 adds direct B metric. |
| E11 | DONE | reports/E11_RHO_SPLIT.md (also run_e11_rho_split.sh) | **seg0 +9.49 pp PASS, align0 +0.97 pp** on 2 FSeval seeds. Gain is dominated by segmentation prior; alignment-only does not pass gate. Seed-2 confirmation in E11c. |
| V02 | DONE | reports/V02_VAOT_PL2_PORT.md | PL 2.x port; all VAOT modules import cleanly |
| E10a | DONE | reports/E10_CURVE_SPACE_COMPARISON.md, run_e10a_bspline_logits.sh | Logits-space B-spline: blend 0.3 −1.11 pp; blend 1.0 −6.21 pp on 2 FSeval seeds; fail gate. |
| E10b | DONE | reports/E10_CURVE_SPACE_COMPARISON.md, run_e10b_bspline_latent.sh | Latent 12-D/blend 0.3 passes 2-seed screening (+2.36 pp); other settings fail. |
| E10c | DONE | reports/E10_CURVE_SPACE_COMPARISON.md, run_e10c_seed2_validation.sh | Seed-2 winner −5.23 pp vs paired baseline; 3-seed mean −0.17 pp and E00 no-collapse gate fails. Scale SKIP. |
| E11c | DONE | reports/E11_RHO_SPLIT.md, run_e11_seed2_validation.sh | Segmentation-only rho=0 gains +7.08 pp on seed 2; 3-seed paired mean +8.69 pp. Advances to FS Mid. |
| E12 | DONE | reports/E12_FS_MID_TRANSFER.md, run_e12_fs_mid_rho_seg0.sh | FS Mid paired seeds: segmentation-only rho=0 gives +0.18 pp mean MoF, mIoU −1.06 pp. Fails gate; DA/YTI scale SKIP. |
| E13 | DONE | reports/E13_ACTION_BRIDGE_RESULTS.md, 4 E13 runs | Cost-only −1.27 pp MoF; bidirectional +0.11 pp and alignment action-agreement proxy +0.88 pp. Mechanically active, fails task gate. |
| E14 | DONE | reports/E14_BSPLINE_FIT_RESULTS.md, 29 fixed-checkpoint audits | All global fits with kmax<256 fail fit gate; kmax≈N is near-identity. Prior curve negatives reclassified as underfit smoothing outcomes. |
| E15a | DONE | reports/E15_BOUNDARY_AWARE_FIT_RESULTS.md, 12 fit audits | Piecewise/repeated-knot-equivalent ceiling fails fit gate even with GT boundaries; ASOT under-segments, posterior changes over-segment. Training stopped. |
| E15b | PENDING | smoothing-spline/Pareto fitter | Replace exact max-error objective with held-out reconstruction vs roughness/capacity selection; linear interpolation control required. |
| E16 | PENDING | | Segment-wise temporal scaling augmentation, gated by reliable pseudo-boundaries. |
| N-A1 | PENDING | | Incremental dynamics-aware structural cost |
| N-A2 | PENDING | | Novel relational distributional transport; gated by N-A1 |
| N-B1 | PENDING | | Incremental transition/phase diagnostics |
| N-B2 | PENDING | | Novel transition-commutative transport; gated by N-B1 |
| N-C1 | PENDING | | Incremental nonlinear temporal-warp diagnostics |
| N-C2 | PENDING | | Novel quotient-space time-reparameterized transport; gated by N-C1 |
| F00 | DONE | reports/F00_FINAL_AUDIT.md | Project-level audit; traceable to manifests/configs/wandb |

## 8. Definition of done toàn dự án

Dự án chỉ được coi là hoàn tất khi:

- DA và YTI có canonical VASOT results, protocol, aggregation và provenance rõ ràng.
- VAOT có ít nhất protocol/data audit và canonical baseline trên mọi dataset khả dụng; dataset không thể chạy phải có blocker được chứng minh.
- A/B/C có controlled diagnostics, effect size và counter-evidence.
- Arc/Gaussian/B-spline/Bézier được so sánh công bằng; không đánh đồng reparameterization với smoothing.
- Final claims tách rõ reproduction, diagnosis, correlation và causal evidence.
- Không có con số nào trong report chỉ tồn tại trong terminal history hoặc một checkpoint không xác định được nguồn.
