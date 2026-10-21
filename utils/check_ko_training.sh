#!/bin/bash
# 한국어 RoboRefit 파인튜닝(roborefit_finetune_ko) 진행 상황 확인용.
# 사용법: bash utils/check_ko_training.sh

OUTDIR="/home/seongyeol/VATEX/results/roborefit_finetune_ko"
METRICS="$OUTDIR/metrics.json"
LOGFILE="$OUTDIR/../roborefit_finetune_ko_nohup.log"

echo "===== 학습 진행 상황 ====="
if [ -f "$METRICS" ]; then
    tail -1 "$METRICS" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
it = d.get('iteration', '?')
print(f'iteration: {it} / 16000')
eta = d.get('eta_seconds')
if eta:
    h, m = int(eta // 3600), int((eta % 3600) // 60)
    print(f'남은 시간: 약 {h}시간 {m}분')
print(f\"total_loss: {d.get('total_loss', 0):.2f}\")
print(f\"loss_ce:    {d.get('loss_ce', 0):.3f}   (final layer, 낮을수록 좋음)\")
print(f\"loss_mask:  {d.get('loss_mask', 0):.3f}\")
print(f\"loss_dice:  {d.get('loss_dice', 0):.3f}\")
print(f\"loss_contrastive: {d.get('loss_contrastive', 0):.3f}\")
print(f\"lr: {d.get('lr')}\")
"
else
    echo "metrics.json 아직 없음 (학습이 아직 시작 전이거나 output dir이 다름)"
fi

echo
echo "===== 프로세스 상태 ====="
PID=$(pgrep -f "train_net_video.py.*swin_base_ko" | head -1)
if [ -n "$PID" ]; then
    ps -p "$PID" -o pid,etime,cmd
else
    echo "!!! 학습 프로세스가 돌고 있지 않습니다 (죽었거나 컴퓨터가 재부팅됐을 수 있음) !!!"
fi

echo
echo "===== GPU 상태 ====="
nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv

echo
echo "===== 저장된 체크포인트 ====="
ls -la "$OUTDIR"/*.pth 2>/dev/null || echo "아직 저장된 체크포인트 없음"

echo
echo "===== 최근 로그 (에러 확인용) ====="
tail -5 "$LOGFILE" 2>/dev/null
