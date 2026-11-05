"""
AI+仿真：激光晶化非晶硅工艺实时寻优系统 (V3.0 - 跨模块联动与数据导出)
"""
import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import io
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. 页面全局配置与状态初始化
# ==========================================
st.set_page_config(page_title="AI+仿真 工业寻优平台", page_icon="⚡", layout="wide")
st.title("⚡ AI+物理仿真：激光晶化工艺智能平台 (V3.0)")
st.markdown("""
**系统简介**：采用百万级数据训练的深度神经网络 (DNN) 替代传统物理仿真，实现 **毫秒级** 工艺参数寻优。
本平台集成了 **实时单点仿真**、**批量数据评估** 与 **AI逆向智能设计** 三大核心业务模块，全面赋能产线工程师。
""")
st.divider()

# 初始化全局记忆变量，用于在 Tab3 和 Tab1 之间传递 Top 5 数据
if 'top_5_recipes' not in st.session_state:
    st.session_state['top_5_recipes'] = None

# ==========================================
# 1. 定义网络与加载预训练资产
# ==========================================
class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_assets():
    model = SurrogateModel()
    model.load_state_dict(torch.load('dnn_proxy_model.pth', map_location='cpu'))
    model.eval()
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_X, scaler_y

try:
    model, scaler_X, scaler_y = load_assets()
except Exception as e:
    st.error(f"模型加载失败，请检查文件: {e}")
    st.stop()

# ==========================================
# 2. 缓存百万级蒙特卡洛预测池
# ==========================================
@st.cache_data(show_spinner=False)
def generate_million_predictions():
    num_samples = 1000000
    E_laser_range = np.random.uniform(0.05, 0.15, num_samples)
    d_film_range = np.random.uniform(30.0, 60.0, num_samples)
    n_val_range = np.random.uniform(3.9, 4.2, num_samples)
    k_val_range = np.random.uniform(0.2, 0.6, num_samples)
    
    X_search = np.column_stack((E_laser_range, d_film_range, n_val_range, k_val_range))
    X_search_scaled = scaler_X.transform(X_search)
    
    with torch.no_grad():
        y_pred_scaled = model(torch.tensor(X_search_scaled, dtype=torch.float32)).numpy()
        
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
    df_results = pd.DataFrame(X_search, columns=['E_laser', 'd_film', 'n', 'k'])
    df_results['T_top'] = y_pred_real[:, 0]
    df_results['T_int'] = y_pred_real[:, 1]
    return df_results

with st.spinner('平台初始化：正在后台加载 1,000,000 次蒙特卡洛数据池，请稍候...'):
    df_pool = generate_million_predictions()

# ==========================================
# 3. 动态漏斗图绘制函数 (新增 Top 5 鲜明标注功能)
# ==========================================
@st.cache_data(show_spinner=False)
def draw_dynamic_funnel_cached(t_top_min, t_int_max, top_5_df=None):
    df_plot_data = df_pool.copy() 
    df_plot_data['is_golden'] = (df_plot_data['T_top'] > t_top_min) & (df_plot_data['T_int'] < t_int_max)
    golden_count = df_plot_data['is_golden'].sum()
    
    df_golden = df_plot_data[df_plot_data['is_golden'] == True]
    df_failed = df_plot_data[df_plot_data['is_golden'] == False]

    plot_golden = df_golden.sample(n=min(len(df_golden), 1500), random_state=42) if len(df_golden) > 0 else df_golden
    plot_failed = df_failed.sample(n=min(len(df_failed), 2500), random_state=42)
    df_plot = pd.concat([plot_failed, plot_golden]).reset_index(drop=True)
    
    cols = ['E_laser', 'd_film', 'n', 'k', 'T_top', 'T_int']
    
    bounds_df = pd.DataFrame([
        [0.05, 30.0, 3.9, 0.2, df_pool['T_top'].min(), df_pool['T_int'].min()],
        [0.15, 60.0, 4.2, 0.6, df_pool['T_top'].max(), df_pool['T_int'].max()]
    ], columns=cols)
    
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(bounds_df)
    df_norm = pd.DataFrame(min_max_scaler.transform(df_plot[cols]), columns=cols)
    df_norm['is_golden'] = df_plot['is_golden'].values

    plt.rcParams.update({'font.family': 'sans-serif', 'axes.linewidth': 1.2})
    fig, ax = plt.subplots(figsize=(14, 5.5))
    x_coords = range(len(cols))

    for i in x_coords: ax.axvline(i, color='black', linewidth=2, zorder=3)

    # 1. 画灰色探索失败配方
    failed_rows = df_norm[df_norm['is_golden'] == False]
    for i in range(len(failed_rows)): ax.plot(x_coords, failed_rows.iloc[i][cols], color='#bdc3c7', alpha=0.05, lw=0.8, zorder=1)

    # 2. 画绿色黄金配方
    golden_rows = df_norm[df_norm['is_golden'] == True]
    for i in range(len(golden_rows)): ax.plot(x_coords, golden_rows.iloc[i][cols], color='#27ae60', alpha=0.3, lw=1.2, zorder=2)

    # 3. 画 AI 逆向推荐的 Top 5 配方 (鲜明高亮: 亮蓝色 DodgerBlue)
    if top_5_df is not None and not top_5_df.empty:
        top_5_norm = pd.DataFrame(min_max_scaler.transform(top_5_df[cols]), columns=cols)
        for i in range(len(top_5_norm)):
            ax.plot(x_coords, top_5_norm.iloc[i][cols], color='#1e90ff', alpha=1.0, lw=3.5, zorder=5)

    dummy_min = pd.DataFrame([[0.05, 30.0, 3.9, 0.2, t_top_min, df_pool['T_int'].min()]], columns=cols)
    dummy_max = pd.DataFrame([[0.15, 60.0, 4.2, 0.6, df_pool['T_top'].max(), t_int_max]], columns=cols)
    norm_min = min_max_scaler.transform(dummy_min)[0]
    norm_max = min_max_scaler.transform(dummy_max)[0]

    rect_top = patches.Rectangle((4 - 0.08, norm_min[4]), 0.16, norm_max[4] - norm_min[4], linewidth=2, edgecolor='#e74c3c', facecolor='#e74c3c', alpha=0.25, zorder=4)
    ax.add_patch(rect_top)
    ax.text(4, norm_max[4] + 0.03, f'> {t_top_min}K', color='#c0392b', fontweight='bold', ha='center')

    rect_int = patches.Rectangle((5 - 0.08, norm_min[5]), 0.16, norm_max[5] - norm_min[5], linewidth=2, edgecolor='#e74c3c', facecolor='#e74c3c', alpha=0.25, zorder=4)
    ax.add_patch(rect_int)
    ax.text(5, norm_max[5] + 0.03, f'< {t_int_max}K', color='#c0392b', fontweight='bold', ha='center')

    ax.set_xticks(x_coords)
    ax.set_xticklabels(['Laser Energy', 'Film Thickness', 'Refractive Index', 'Extinction Coeff', 'Top Temp', 'Interface Temp'], fontweight='bold', fontsize=11)
    ax.set_yticks([])
    ax.set_ylim(-0.05, 1.15) 
    
    # 动态组装图例
    import matplotlib.lines as mlines
    l1 = mlines.Line2D([], [], color='#bdc3c7', lw=2, label='Explored Space')
    l2 = mlines.Line2D([], [], color='#27ae60', lw=2, label=f'Golden Recipes ({golden_count} found)')
    l3 = patches.Patch(facecolor='#e74c3c', edgecolor='#e74c3c', alpha=0.3, label='Target Window')
    handles_list = [l1, l2, l3]
    
    if top_5_df is not None and not top_5_df.empty:
        l4 = mlines.Line2D([], [], color='#1e90ff', lw=3.5, label='Top 5 Recommended (from Tab 3)')
        handles_list.append(l4)

    ax.legend(handles=handles_list, loc='upper right', frameon=False, fontsize=10)
    sns.despine(top=True, right=True, bottom=True, left=True)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    plt.close(fig) 
    return buf.getvalue(), golden_count

# ==========================================
# 4. 构建三大业务工作台 (多标签页结构)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 实时单点仿真", "📂 批量工艺评估", "🎯 AI 逆向智能推荐"])

# ------------------------------------------
# 模块一：实时单点仿真 (含漏斗图导出)
# ------------------------------------------
with tab1:
    st.subheader("1. 基础工艺配方输入")
    col1, col2, col3, col4 = st.columns(4)
    E_laser = col1.slider("激光能量 (E_laser)", 0.05, 0.15, 0.10, 0.001,format="%.4f")
    d_film = col2.slider("膜层厚度 (d_film)", 30.0, 60.0, 45.0, 0.5)
    n_val = col3.slider("折射率 (n)", 3.9, 4.2, 4.05, 0.01)
    k_val = col4.slider("消光系数 (k)", 0.2, 0.6, 0.4, 0.01)

    X_input = np.array([[E_laser, d_film, n_val, k_val]])
    y_pred = scaler_y.inverse_transform(model(torch.tensor(scaler_X.transform(X_input), dtype=torch.float32)).detach().numpy())[0]
    T_top_pred, T_int_pred = y_pred[0], y_pred[1]

    st.subheader("2. 靶心目标与动态漏斗图")
    t_col1, t_col2 = st.columns(2)
    target_T_top = t_col1.slider("T_top 目标下限 (K)", 1250, 1450, 1350, 10, key="t1")
    target_T_int = t_col2.slider("T_int 目标上限 (K)", 1250, 1450, 1380, 10, key="t2")

    is_T_top_ok = T_top_pred > target_T_top
    is_T_int_ok = T_int_pred < target_T_int

    st.markdown("##### 实时预测仪表盘")
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("顶层峰值温度 (T_top)", f"{T_top_pred:.1f} K", f"Target: > {target_T_top} K", delta_color="normal" if is_T_top_ok else "inverse")
    m_col2.metric("界面温度 (T_int)", f"{T_int_pred:.1f} K", f"Target: < {target_T_int} K", delta_color="normal" if is_T_int_ok else "inverse")

    if is_T_top_ok and is_T_int_ok:
        st.success("✅ **寻优成功！** 当前工艺参数落入黄金工艺窗口。")
    else:
        st.error("⚠️ **参数不达标！** 当前温度超出安全窗口。")

    # 绘制带联动标注的漏斗图
    img_bytes, found_count = draw_dynamic_funnel_cached(target_T_top, target_T_int, st.session_state['top_5_recipes'])
    st.image(img_bytes, use_container_width=True)

    # 提取当前靶心过滤后的全部合格数据，提供下载接口
    st.markdown("---")
    st.markdown(f"##### 📥 黄金工艺数据提取接口")
    df_golden_export = df_pool[(df_pool['T_top'] > target_T_top) & (df_pool['T_int'] < target_T_int)]
    st.download_button(
        label=f"💾 导出漏斗图当前圈定的 {found_count} 组黄金配方至 CSV 报表",
        data=df_golden_export.to_csv(index=False).encode('utf-8-sig'),
        file_name="Filtered_Golden_Recipes.csv",
        mime="text/csv",
        type="primary"
    )

# ------------------------------------------
# 模块二：批量工艺评估 
# ------------------------------------------
with tab2:
    st.subheader("📂 历史配方批量秒级评估")
    st.markdown("上传包含 `E_laser`, `d_film`, `n`, `k` 列的 CSV 表格。AI 将在一秒内并行完成全部物理仿真，并高亮异常配方。")
    
    template_df = pd.DataFrame({"E_laser": [0.10, 0.12, 0.08], "d_film": [45.0, 50.0, 35.0], "n": [4.05, 4.10, 3.95], "k": [0.4, 0.3, 0.5]})
    st.download_button("📥 下载测试模板 (CSV)", template_df.to_csv(index=False).encode('utf-8'), "Template.csv", "text/csv")
    
    uploaded_file = st.file_uploader("请上传待评估的工艺文件 (CSV 格式)", type=['csv'])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        if set(['E_laser', 'd_film', 'n', 'k']).issubset(batch_df.columns):
            with st.spinner('AI 正在高速推演温度场...'):
                X_batch = scaler_X.transform(batch_df[['E_laser', 'd_film', 'n', 'k']])
                y_batch_pred = scaler_y.inverse_transform(model(torch.tensor(X_batch, dtype=torch.float32)).detach().numpy())
                
                batch_df['T_top_预测值'] = y_batch_pred[:, 0].round(1)
                batch_df['T_int_预测值'] = y_batch_pred[:, 1].round(1)
                batch_df['工艺判定'] = (batch_df['T_top_预测值'] > 1350) & (batch_df['T_int_预测值'] < 1380)
                batch_df['工艺判定'] = batch_df['工艺判定'].map({True: '✅ 合格 (黄金窗口)', False: '❌ 不合格 (报废风险)'})

                def highlight_status(row):
                    color = '#d4edda' if '合格' in row['工艺判定'] else '#f8d7da'
                    return [f'background-color: {color}'] * len(row)
                
                st.markdown("##### 📝 批量评估报告")
                st.dataframe(batch_df.style.apply(highlight_status, axis=1), use_container_width=True)
                st.download_button("📤 导出评估报告 (CSV)", batch_df.to_csv(index=False).encode('utf-8-sig'), "AI_Evaluation_Report.csv", "text/csv")
        else:
            st.error("表格格式错误！必须包含列：E_laser, d_film, n, k")

# ------------------------------------------
# 模块三：AI 逆向智能推荐
# ------------------------------------------
with tab3:
    st.subheader("🎯 专家级 AI 逆向配方推荐 (Inverse Design)")
    st.markdown("输入**最终期望的物理温度目标**，AI 引擎将瞬间在 1,000,000 组参数空间中为您锁定匹配度最高的黄金工艺组合。")
    
    r_col1, r_col2 = st.columns(2)
    ideal_T_top = r_col1.number_input("您的理想 T_top (K)", value=1365.0, step=1.0)
    ideal_T_int = r_col2.number_input("您的理想 T_int (K)", value=1310.0, step=1.0)
    
    if st.button("🚀 启动 AI 寻优检索", type="primary"):
        with st.spinner('正在从 1,000,000 组工艺空间中进行欧式距离匹配检索...'):
            df_search = df_pool.copy()
            df_search['偏差得分'] = (df_search['T_top'] - ideal_T_top)**2 + (df_search['T_int'] - ideal_T_int)**2
            
            # 提取 Top 5 并在展示前存入全局状态，供 Tab 1 的漏斗图读取高亮
            top_5 = df_search.sort_values(by='偏差得分').head(5).reset_index(drop=True)
            st.session_state['top_5_recipes'] = top_5.copy()
            
            top_5['T_top'] = top_5['T_top'].round(1)
            top_5['T_int'] = top_5['T_int'].round(1)
            top_5['E_laser'] = top_5['E_laser'].round(4)
            top_5['d_film'] = top_5['d_film'].round(2)
            
            st.success("✅ 检索完毕！已为您匹配到当前目标下最优的 5 组工艺配方。您可以返回 **[Tab1: 实时单点仿真]** 查看它们在参数空间中的整体分布位置！")
            
            for i in range(5):
                recipe = top_5.iloc[i]
                with st.expander(f"🏅 Top {i+1} 推荐配方 (偏差最小) - 点击查看参数", expanded=(i==0)):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("推荐 E_laser", recipe['E_laser'])
                    c2.metric("推荐 d_film", recipe['d_film'])
                    c3.metric("推荐 n", recipe['n'].round(3))
                    c4.metric("推荐 k", recipe['k'].round(3))
                    st.caption(f"*该配方预测温度：T_top={recipe['T_top']} K, T_int={recipe['T_int']} K*")
