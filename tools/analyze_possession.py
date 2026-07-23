"""
追踪帧102后完整控球链分析脚本
"""
import csv
import math
from collections import defaultdict

# ============================================================
# 1. 读取数据
# ============================================================

def read_ball_csv(filepath):
    """读取球3D坐标CSV → {frame: (x, y, z)}"""
    ball = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = int(row['frame_num'])
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z'])
            ball[fn] = (x, y, z)
    return ball

def read_person_csv(filepath):
    """读取球员2D坐标CSV → list of (frame, track_id, x, y, color)"""
    persons = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = int(row['frame_num'])
            tid = int(row['track_id'])
            x = float(row['x'])
            y = float(row['y'])
            color = row['color']
            persons.append((fn, tid, x, y, color))
    return persons

ball_data = read_ball_csv(r'd:\GHY\2.Files\football-agent\TestInput\Files\12s_soccer3d.csv')
person_data = read_person_csv(r'd:\GHY\2.Files\football-agent\TestInput\Files\12s_person2d.csv')

# ============================================================
# 2. 建立球员最后已知位置追踪
# ============================================================

# 所有出现的帧
all_frames = sorted(set([fn for fn,_,_,_,_ in person_data] + list(ball_data.keys())))

# 为每个球员追踪最后位置: {frame: {player_id: (x, y, color)}}
player_pos_at_frame = {}
current_positions = {}

# 按帧排序所有person数据
person_by_frame = defaultdict(list)
for fn, tid, x, y, color in person_data:
    person_by_frame[fn].append((tid, x, y, color))

for fn in sorted(person_by_frame.keys()):
    # 更新当前位置
    for tid, x, y, color in person_by_frame[fn]:
        current_positions[tid] = (x, y, color)
    # 快照
    player_pos_at_frame[fn] = dict(current_positions)

# 对于中间帧（无球员更新），向前查找最近已知位置
def get_player_positions_at_frame(target_frame):
    """获取某帧所有球员位置（向前查找）"""
    # 找到 <= target_frame 的最大帧
    available = [f for f in sorted(player_pos_at_frame.keys()) if f <= target_frame]
    if not available:
        return {}
    return player_pos_at_frame[available[-1]]

# ============================================================
# 3. 帧102后的球帧
# ============================================================

ball_frames_after_102 = [f for f in sorted(ball_data.keys()) if f >= 102]
print(f"帧102后的球帧共 {len(ball_frames_after_102)} 帧:")
print(ball_frames_after_102)

# ============================================================
# 4. 计算距离
# ============================================================

def dist_2d(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# ============================================================
# 5. 为每个球帧找最近球员
# ============================================================

print("\n" + "="*100)
print("每球帧的最近球员分析")
print("="*100)

# 所有球员ID
all_pids = sorted(set(tid for _, tid, _, _, _ in person_data))

for bf in ball_frames_after_102:
    bx, by, bz = ball_data[bf]
    players = get_player_positions_at_frame(bf)
    
    # 计算所有球员距离
    distances = []
    for pid in all_pids:
        if pid in players:
            px, py, color = players[pid]
            d = dist_2d((bx, by), (px, py))
            distances.append((d, pid, color, px, py))
    
    distances.sort()
    
    # 最近5个
    print(f"\n帧 {bf}: 球({bx:.1f}, {by:.1f}, {bz:.1f})")
    for i, (d, pid, color, px, py) in enumerate(distances[:5]):
        marker = " ★控球" if i == 0 else ""
        print(f"  #{i+1}: {pid}号({color}队) 距={d:.1f}px 位置({px:.1f}, {py:.1f}){marker}")

# ============================================================
# 6. 相邻帧对分析
# ============================================================

print("\n" + "="*100)
print("相邻球帧对分析 (距离 / 速度 / 球员变化)")
print("="*100)

# FPS = 30
FPS = 30

for i in range(len(ball_frames_after_102) - 1):
    f1 = ball_frames_after_102[i]
    f2 = ball_frames_after_102[i + 1]
    
    b1 = ball_data[f1]
    b2 = ball_data[f2]
    d = dist_2d((b1[0], b1[1]), (b2[0], b2[1]))
    frame_diff = f2 - f1
    speed = d * FPS / frame_diff if frame_diff > 0 else 0  # px/s
    speed_ms = d * 30 / frame_diff  # same calculation
    
    # 最近球员
    p1 = get_player_positions_at_frame(f1)
    p2 = get_player_positions_at_frame(f2)
    
    # 找出每帧最近的3个球员
    d1_list = []
    for pid in all_pids:
        if pid in p1:
            px, py, c = p1[pid]
            d1_list.append((dist_2d((b1[0], b1[1]), (px, py)), pid, c))
    d1_list.sort()
    
    d2_list = []
    for pid in all_pids:
        if pid in p2:
            px, py, c = p2[pid]
            d2_list.append((dist_2d((b2[0], b2[1]), (px, py)), pid, c))
    d2_list.sort()
    
    top1_start = d1_list[0]
    top1_end = d2_list[0]
    player_changed = top1_start[1] != top1_end[1]
    
    status = "静止" if d < 1 else ("慢移" if d < 20 else ("移动" if d < 80 else "快速移动"))
    
    print(f"\n帧 {f1}→{f2} (Δ={frame_diff}帧):")
    print(f"  球: ({b1[0]:.1f},{b1[1]:.1f}) → ({b2[0]:.1f},{b2[1]:.1f})")
    print(f"  距离={d:.1f}px | 速度={speed:.1f}px/s | {status}")
    print(f"  最近球员: {top1_start[1]}号({top1_start[2]}队) [距{top1_start[0]:.1f}] → {top1_end[1]}号({top1_end[2]}队) [距{top1_end[0]:.1f}]")
    if player_changed:
        # 检查起始帧最近球员是否在结束帧附近
        start_pid = top1_start[1]
        if start_pid in p2:
            d_check = dist_2d((b2[0], b2[1]), (p2[start_pid][0], p2[start_pid][1]))
            print(f"  ⚠ 球员变化! 原控球{start_pid}号距新球位={d_check:.1f}px")
        else:
            print(f"  ⚠ 球员变化! 原控球{start_pid}号在帧{f2}无数据")
    
    # 检查球附近是否有球员（可疑）
    suspicious = []
    for d, pid, c in d1_list[:5]:
        if d > 100:
            suspicious.append(f"帧{f1}: {pid}号距球{d:.0f}px")
    for d, pid, c in d2_list[:5]:
        if d > 100:
            suspicious.append(f"帧{f2}: {pid}号距球{d:.0f}px")
    if suspicious:
        print(f"  ⚠ 可疑: {', '.join(suspicious)}")

# ============================================================
# 7. 分段总结
# ============================================================

print("\n" + "="*100)
print("控球分段总结")
print("="*100)

segments = []
current_segment = {
    'start_frame': ball_frames_after_102[0],
    'start_pos': ball_data[ball_frames_after_102[0]],
    'nearest_start': None,
}

for i in range(len(ball_frames_after_102) - 1):
    f1 = ball_frames_after_102[i]
    f2 = ball_frames_after_102[i + 1]
    
    b1 = ball_data[f1]
    b2 = ball_data[f2]
    d = dist_2d((b1[0], b1[1]), (b2[0], b2[1]))
    
    p1 = get_player_positions_at_frame(f1)
    d1_list = []
    for pid in all_pids:
        if pid in p1:
            px, py, c = p1[pid]
            d1_list.append((dist_2d((b1[0], b1[1]), (px, py)), pid, c))
    d1_list.sort()
    
    p2 = get_player_positions_at_frame(f2)
    d2_list = []
    for pid in all_pids:
        if pid in p2:
            px, py, c = p2[pid]
            d2_list.append((dist_2d((b2[0], b2[1]), (px, py)), pid, c))
    d2_list.sort()
    
    if current_segment['nearest_start'] is None:
        current_segment['nearest_start'] = d1_list[0]
    
    player_changed = d1_list[0][1] != d2_list[0][1] if d1_list and d2_list else False
    
    # 决定是否开始新段：球员变了 或 速度大变化
    gap = f2 - f1
    speed = d * FPS / gap if gap > 0 else 0
    
    frame_diff_large = gap > 20
    
    if player_changed or frame_diff_large or d > 80:
        segments.append({
            **current_segment,
            'end_frame': f1,
            'end_pos': b1,
            'nearest_end': d1_list[0],
            'frames': f"{current_segment['start_frame']}→{f1}",
            'distance': d,
        })
        current_segment = {
            'start_frame': f2,
            'start_pos': b2,
            'nearest_start': d2_list[0],
        }

# 添加最后一段
last_f = ball_frames_after_102[-1]
p_last = get_player_positions_at_frame(last_f)
dl = []
for pid in all_pids:
    if pid in p_last:
        px, py, c = p_last[pid]
        dl.append((dist_2d((ball_data[last_f][0], ball_data[last_f][1]), (px, py)), pid, c))
dl.sort()

segments.append({
    **current_segment,
    'end_frame': last_f,
    'end_pos': ball_data[last_f],
    'nearest_end': dl[0] if dl else None,
    'frames': f"{current_segment['start_frame']}→{last_f}",
})

for seg in segments:
    ns = seg.get('nearest_start')
    ne = seg.get('nearest_end')
    start_pid = f"{ns[1]}号({ns[2]}队)" if ns else "?"
    end_pid = f"{ne[1]}号({ne[2]}队)" if ne else "?"
    
    b_start = seg['start_pos']
    b_end = seg['end_pos']
    total_dist = dist_2d((b_start[0], b_start[1]), (b_end[0], b_end[1]))
    frame_span = seg['end_frame'] - seg['start_frame']
    avg_speed = total_dist * FPS / frame_span if frame_span > 0 else 0
    
    if total_dist < 1:
        segment_type = "静止"
    elif total_dist < 20:
        segment_type = "轻微移动(可能带球)"
    elif total_dist < 80:
        segment_type = "缓慢移动(可能带球)"
    else:
        segment_type = "快速移动(传球/解围)"
    
    print(f"\n段: {seg['frames']} (跨{frame_span}帧)")
    print(f"  球: ({b_start[0]:.1f},{b_start[1]:.1f},{b_start[2]:.1f}) → ({b_end[0]:.1f},{b_end[1]:.1f},{b_end[2]:.1f})")
    print(f"  总距离={total_dist:.1f}px | 均速={avg_speed:.1f}px/s | 类型={segment_type}")
    print(f"  最近球员: {start_pid} → {end_pid}")

# ============================================================
# 8. 重点关注区域分析
# ============================================================

print("\n" + "="*100)
print("重点区域分析")
print("="*100)

# A. 帧102-123: 球静止在(871.4, 559.9)
print("\n--- A. 帧102→123 球静止期 (871.4, 559.9) ---")
for fn in [102, 112, 122, 123]:
    bx, by, bz = ball_data[fn]
    players = get_player_positions_at_frame(fn)
    dists = []
    for pid in all_pids:
        if pid in players:
            px, py, c = players[pid]
            d = dist_2d((bx, by), (px, py))
            dists.append((d, pid, c, px, py))
    dists.sort()
    print(f"  帧{fn}: 球({bx:.1f},{by:.1f})")
    for d, pid, c, px, py in dists[:5]:
        print(f"    {pid}号({c}队) 距{d:.1f}px 位置({px:.1f},{py:.1f})")

# B. 帧123→149: 球从(871.4,559.9)→(977.3,669)
print("\n--- B. 帧123→149 球移动 (871.4,559.9)→(977.3,669) ---")
b123 = ball_data[123]
b149 = ball_data[149]
d_123_149 = dist_2d((b123[0], b123[1]), (b149[0], b149[1]))
speed_123_149 = d_123_149 * 30 / (149-123)
print(f"  距离={d_123_149:.1f}px | 速度={speed_123_149:.1f}px/s (跨26帧)")
# 查帧123和149附近的球员
for fn in [123, 149]:
    bx, by, bz = ball_data[fn]
    players = get_player_positions_at_frame(fn)
    dists = []
    for pid in all_pids:
        if pid in players:
            px, py, c = players[pid]
            d = dist_2d((bx, by), (px, py))
            dists.append((d, pid, c, px, py))
    dists.sort()
    print(f"  帧{fn}:")
    for d, pid, c, px, py in dists[:5]:
        print(f"    {pid}号({c}队) 距{d:.1f}px 位置({px:.1f},{py:.1f})")

# C. 帧149→223: 6号附近
print("\n--- C. 帧149→223 6号控球期 ---")
for fn in ball_frames_after_102:
    if 149 <= fn <= 223:
        bx, by, bz = ball_data[fn]
        players = get_player_positions_at_frame(fn)
        dists = []
        for pid in all_pids:
            if pid in players:
                px, py, c = players[pid]
                d = dist_2d((bx, by), (px, py))
                dists.append((d, pid, c, px, py))
        dists.sort()
        # 只关心前3
        top3 = [f"{pid}号({c}队)距{d:.0f}" for d, pid, c, px, py in dists[:3]]
        print(f"  帧{fn}: 球({bx:.1f},{by:.1f},{bz:.1f}) | 最近: {', '.join(top3)}")

# D. 帧223→349: 剩余追踪
print("\n--- D. 帧223→349 后续事件 ---")
for fn in ball_frames_after_102:
    if fn >= 223:
        bx, by, bz = ball_data[fn]
        players = get_player_positions_at_frame(fn)
        dists = []
        for pid in all_pids:
            if pid in players:
                px, py, c = players[pid]
                d = dist_2d((bx, by), (px, py))
                dists.append((d, pid, c, px, py))
        dists.sort()
        top3 = [f"{pid}号({c}队)距{d:.0f}" for d, pid, c, px, py in dists[:3]]
        print(f"  帧{fn}: 球({bx:.1f},{by:.1f},{bz:.1f}) | 最近: {', '.join(top3)}")

# E. 可疑点汇总
print("\n--- E. 可疑时段 ---")
for fn in ball_frames_after_102:
    bx, by, bz = ball_data[fn]
    players = get_player_positions_at_frame(fn)
    dists = []
    for pid in all_pids:
        if pid in players:
            px, py, c = players[pid]
            d = dist_2d((bx, by), (px, py))
            dists.append((d, pid, c, px, py))
    dists.sort()
    if dists and dists[0][0] > 50:
        print(f"  ⚠ 帧{fn}: 最近球员{dists[0][1]}号距{dists[0][0]:.1f}px > 50px")

# F. 帧232球突然跳回(1066.8, 380.5)
print("\n--- F. 帧223→232 球突然跳回 ---")
print(f"  帧223: ({ball_data[223][0]:.1f}, {ball_data[223][1]:.1f})")
print(f"  帧232: ({ball_data[232][0]:.1f}, {ball_data[232][1]:.1f})")
d_jump = dist_2d((ball_data[223][0], ball_data[223][1]), (ball_data[232][0], ball_data[232][1]))
speed_jump = d_jump * 30 / 9
print(f"  距离={d_jump:.1f}px | 速度={speed_jump:.1f}px/s (跨9帧)")
# 查帧232附近球员
for fn in [223, 232]:
    bx, by, bz = ball_data[fn]
    players = get_player_positions_at_frame(fn)
    dists = []
    for pid in all_pids:
        if pid in players:
            px, py, c = players[pid]
            d = dist_2d((bx, by), (px, py))
            dists.append((d, pid, c, px, py))
    dists.sort()
    print(f"  帧{fn} 最近:")
    for d, pid, c, px, py in dists[:3]:
        print(f"    {pid}号({c}队) 距{d:.1f}px")

print("\n分析完成!")
