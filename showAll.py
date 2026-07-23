import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import platform
import matplotlib.font_manager as fm

def set_chinese_font():
    system = platform.system()
    if system == 'Windows':
        font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong']
    elif system == 'Darwin':  
        font_list = ['Heiti SC', 'STHeiti', 'Arial Unicode MS']
    else:  
        font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_found = False
    for font in font_list:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            font_found = True
            break
    if not font_found:
        print("未找到合适的中文字体，使用备用方案...")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False 

set_chinese_font()

# 读取球员数据
person_file = "12s_cf_f0_person.csv"
print(u"正在读取球员数据: {}".format(person_file))
df_person_raw = pd.read_csv(person_file)

# 读取足球数据
ball_file = "12s_cf_f0_ball.csv"
print(u"正在读取足球数据: {}".format(ball_file))
df_ball_raw = pd.read_csv(ball_file)

# 获取帧范围（取球员和足球的交集）
frame_min = max(df_person_raw['frame_num'].min(), df_ball_raw['frame_num'].min())
frame_max = min(df_person_raw['frame_num'].max(), df_ball_raw['frame_num'].max())
all_frames = np.arange(frame_min, frame_max + 1)

# 插值球员数据
all_track_ids = df_person_raw['track_id'].unique()
interp_list = []
for track_id in all_track_ids:
    track = df_person_raw[df_person_raw['track_id'] == track_id].sort_values('frame_num')
    interp_x = np.interp(all_frames, track['frame_num'], track['x'])
    interp_y = np.interp(all_frames, track['frame_num'], track['y'])
    color_arr = track['color'].values
    color_idx = np.searchsorted(track['frame_num'], all_frames, side='right') - 1
    color_idx = np.clip(color_idx, 0, len(color_arr)-1)
    interp_color = color_arr[color_idx]
    interp_df = pd.DataFrame({
        'frame_num': all_frames,
        'track_id': track_id,
        'x': interp_x,
        'y': interp_y,
        'color': interp_color
    })
    interp_list.append(interp_df)
df_person = pd.concat(interp_list, ignore_index=True)

# 插值足球数据
interp_ball_x = np.interp(all_frames, df_ball_raw['frame_num'], df_ball_raw['x'])
interp_ball_y = np.interp(all_frames, df_ball_raw['frame_num'], df_ball_raw['y'])
df_ball = pd.DataFrame({
    'frame_num': all_frames,
    'x': interp_ball_x,
    'y': interp_ball_y
})

# 创建图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
plt.subplots_adjust(bottom=0.25)

initial_frame = frame_min
ax_frame = plt.axes([0.25, 0.1, 0.65, 0.03])
frame_slider = Slider(
    ax=ax_frame,
    label=u'帧数',
    valmin=frame_min,
    valmax=frame_max,
    valinit=initial_frame,
    valstep=1
)

track_lines = {}
current_points = {}
ball_line = None
ball_point = None

colors = {
    'A': 'red',
    'B': 'blue',
    'C': 'green',
}

def update_plot(frame):
    global track_lines, current_points, ball_line, ball_point
    
    # 清除球员轨迹和点
    for line in track_lines.values():
        line.remove()
    for point in current_points.values():
        point.remove()
    for text in ax.texts[:]:
        text.remove()
    track_lines = {}
    current_points = {}
    
    # 清除足球轨迹和点
    if ball_line:
        ball_line.remove()
    if ball_point:
        ball_point.remove()
    
    # 绘制球员轨迹和当前位置
    for track_id in df_person['track_id'].unique():
        track_data = df_person[df_person['track_id'] == track_id]
        color = colors.get(track_data.iloc[0]['color'], 'gray')
        traj = track_data[track_data['frame_num'] <= frame]
        if len(traj) > 1:
            line, = ax.plot(traj['x'], traj['y'], '-', color=color, alpha=0.5, linewidth=1.5)
            track_lines[track_id] = line
        current = track_data[track_data['frame_num'] == frame]
        if not current.empty:
            point = ax.scatter(current['x'], current['y'], c=color, s=60, label=f'球员 {track_id}', zorder=5)
            current_points[track_id] = point
            # 添加球员序号标签
            x_pos = current['x'].values[0]
            y_pos = current['y'].values[0]
            ax.text(x_pos, y_pos, str(track_id), fontsize=8, ha='center', va='center', 
                   color='white', weight='bold', bbox=dict(boxstyle='circle', facecolor=color, alpha=0.8), zorder=6)
    
    # 绘制足球轨迹和当前位置
    ball_traj = df_ball[df_ball['frame_num'] <= frame]
    if len(ball_traj) > 1:
        ball_line, = ax.plot(ball_traj['x'], ball_traj['y'], '-', color='white', alpha=0.8, linewidth=2.5, label=u'足球轨迹', zorder=7)
    ball_current = df_ball[df_ball['frame_num'] == frame]
    if not ball_current.empty:
        ball_point = ax.scatter(ball_current['x'], ball_current['y'], c='white', s=100, marker='o', 
                               edgecolors='black', linewidths=2, label=u'足球', zorder=8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(u'足球和球员轨迹 - 帧 {}'.format(frame))
    ax.set_xlim(-60, 1260)   # 扩展边界，让界外球/发球点可见
    ax.set_ylim(-60, 760)
    ax.set_facecolor('#2d5016')  # 设置草地绿色背景
    
    # 更新图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if ax.get_legend():
        ax.get_legend().remove()
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    
    fig.canvas.draw_idle()

def update(val):
    frame = int(frame_slider.val)
    update_plot(frame)

frame_slider.on_changed(update)

update_plot(initial_frame)

plt.show()
