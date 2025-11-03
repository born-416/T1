import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import base64

warnings.filterwarnings('ignore')

# 设置页面
st.set_page_config(
    page_title="T1 2024世界赛夺冠深度分析",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式 - 黑红色系优化（不依赖外部字体）
st.markdown("""
<style>
.main-header {
    font-size: 3.5rem;
    background: linear-gradient(135deg, #E2012D 0%, #8B0000 50%, #000000 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-weight: 900;
    margin-bottom: 2rem;
    text-shadow: 0 0 30px rgba(226, 1, 45, 0.3);
    letter-spacing: 2px;
    font-family: Arial, sans-serif;
}

.section-header {
    font-size: 2.2rem;
    color: #E2012D;
    border-bottom: 3px solid #E2012D;
    padding-bottom: 0.8rem;
    margin-top: 2.5rem;
    margin-bottom: 1.5rem;
    font-weight: 700;
    text-shadow: 0 0 10px rgba(226, 1, 45, 0.3);
    font-family: Arial, sans-serif;
}

.subsection-header {
    font-size: 1.6rem;
    color: #E2012D;
    border-left: 5px solid #E2012D;
    padding-left: 1.2rem;
    margin-top: 2rem;
    margin-bottom: 1.2rem;
    font-weight: 600;
    font-family: Arial, sans-serif;
}

.metric-card {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    padding: 1.8rem;
    border-radius: 15px;
    border: 2px solid #E2012D;
    margin: 0.8rem 0;
    box-shadow: 0 8px 25px rgba(226, 1, 45, 0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 35px rgba(226, 1, 45, 0.3);
}

.t1-player-card {
    background: linear-gradient(135deg, #1a0000 0%, #330000 50%, #1a1a1a 100%);
    padding: 1.5rem;
    border-radius: 15px;
    border: 2px solid #E2012D;
    margin: 1rem 0;
    box-shadow: 0 6px 20px rgba(226, 1, 45, 0.2);
    transition: all 0.3s ease;
}

.t1-player-card:hover {
    transform: translateX(10px);
    box-shadow: 0 10px 30px rgba(226, 1, 45, 0.4);
}

.player-profile {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    border-radius: 20px;
    padding: 2rem;
    border: 2px solid #E2012D;
    box-shadow: 0 10px 30px rgba(226, 1, 45, 0.2);
    margin: 1.5rem 0;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #E2012D, #8B0000);
}

/* 侧边栏样式 */
.css-1d391kg, .css-1lcbmhc {
    background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%) !important;
}

/* 音乐播放器样式 */
.music-player {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: rgba(26, 0, 0, 0.9);
    border: 2px solid #E2012D;
    border-radius: 25px;
    padding: 15px;
    box-shadow: 0 0 30px rgba(226, 1, 45, 0.5);
    backdrop-filter: blur(10px);
    z-index: 1000;
}

.music-info {
    color: #E2012D;
    font-size: 0.9rem;
    text-align: center;
    margin-bottom: 8px;
    font-weight: bold;
}

/* 标签页样式 */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: #1a1a1a;
    border-radius: 10px 10px 0 0;
    padding: 12px 24px;
    border: 1px solid #333;
    color: #888;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: #2a0000;
    color: #E2012D;
    border-color: #E2012D;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #E2012D 0%, #8B0000 100%) !important;
    color: white !important;
    border-color: #E2012D !important;
}

/* 数据表格样式 */
.dataframe {
    background: #1a1a1a !important;
    color: white !important;
}

/* 按钮样式 */
.stButton button {
    background: linear-gradient(135deg, #E2012D 0%, #8B0000 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(226, 1, 45, 0.4);
}

/* 选择框样式 */
.stSelectbox div div {
    background: #1a1a1a;
    border: 1px solid #333;
    color: white;
}

/* 滚动条样式 */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #1a1a1a;
}

::-webkit-scrollbar-thumb {
    background: #E2012D;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #8B0000;
}

/* 通用文本样式 */
body {
    color: #e0e0e0;
    background-color: #0a0a0a;
}
</style>
""", unsafe_allow_html=True)


# 背景音乐函数
def autoplay_audio(file_path: str):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
            <div class="music-player">
                <div class="music-info">🎵 Legends Never Die</div>
                <audio controls autoplay loop style="width: 200px;">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
            </div>
            """
            st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.warning(f"背景音乐加载失败: {str(e)}")


# 加载数据
@st.cache_data
def load_data():
    # 这里使用示例数据，请替换为您的实际数据路径
    try:
        player_df = pd.read_csv("player_statistics_cleaned_final.csv")
        champions_df = pd.read_csv("champions.csv")
    except:
        # 如果文件不存在，创建示例数据
        player_data = {
            'PlayerName': ['Zeus', 'Oner', 'Faker', 'Gumayusi', 'Keria'],
            'TeamName': ['T1', 'T1', 'T1', 'T1', 'T1'],
            'Position': ['Top', 'Jungle', 'Mid', 'Adc', 'Support'],
            'Win rate': [0.875, 0.72, 0.78, 0.74, 0.73],
            'KDA': [9.10, 3.8, 5.1, 4.8, 4.0],
            'DPM': [696, 320, 580, 620, 180],
            'KP%': [0.716, 0.75, 0.68, 0.72, 0.78],
            'GoldPerMin': [380, 320, 420, 450, 280],
            'GD@15': [350, 280, 420, 380, 150],
            'Games': [25, 25, 25, 25, 25]
        }
        player_df = pd.DataFrame(player_data)

        champions_data = {
            'Champion': ['Ahri', 'Aatrox', 'Aphelios', 'Leona', 'Lee Sin'],
            'Picks': [45, 38, 42, 35, 40],
            'Bans': [30, 25, 35, 20, 28],
            'Winrate': [0.52, 0.48, 0.55, 0.58, 0.50],
            'Presence': [0.75, 0.63, 0.77, 0.55, 0.68],
            'KDA': [3.2, 2.8, 3.5, 3.0, 3.1]
        }
        champions_df = pd.DataFrame(champions_data)

    return player_df, champions_df


player_df, champions_df = load_data()

# 颜色配置
T1_RED = '#E2012D'
T1_BLACK = '#000000'
POSITION_COLORS = {
    'Top': '#1f77b4',
    'Jungle': '#2ca02c',
    'Mid': '#ff7f0e',
    'Adc': '#d62728',
    'Support': '#9467bd'
}

# 选手信息（包含简介和图片路径）
PLAYER_INFO = {
    "zeus": {
        "name": "Zeus",
        "real_name": "Choi Woo-je",
        "position": "Top",
        "description": "世界最强上单之一，以出色的对线能力和英雄池著称。2024世界赛FMVP，以其惊人的操作和团战表现带领T1走向胜利。",
        "image_url": "zeus.jpg"
    },
    "oner": {
        "name": "Oner",
        "real_name": "Moon Hyeon-joon",
        "position": "Jungle",
        "description": "侵略性极强的打野选手，以其精准的节奏控制和地图掌控能力闻名。在关键比赛中总能找到最佳开团时机。",
        "image_url": "oner.jpg"
    },
    "faker": {
        "name": "Faker",
        "real_name": "Lee Sang-hyeok",
        "position": "Mid",
        "description": "英雄联盟历史上最伟大的选手，五冠王得主。以其无与伦比的游戏理解、领导力和关键时刻的carry能力著称。",
        "image_url": "faker.jpg"
    },
    "gumayusi": {
        "name": "Gumayusi",
        "real_name": "Lee Min-hyeong",
        "position": "ADC",
        "description": "世界顶级ADC选手，以其稳定的输出和极限操作闻名。在团战中总能找到最佳输出位置，是T1的可靠后期保障。",
        "image_url": "gumayusi.jpg"
    },
    "keria": {
        "name": "Keria",
        "real_name": "Ryu Min-seok",
        "position": "Support",
        "description": "天才辅助选手，以其创新的玩法和精准的开团能力著称。被认为是世界上最具创造力的辅助选手。",
        "image_url": "keria.jpg"
    }
}

# 主页标题
st.markdown('<div class="main-header">🏆 T1 2024英雄联盟世界赛夺冠深度分析</div>', unsafe_allow_html=True)

# 侧边栏
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #1a0000 0%, #330000 100%); border-radius: 10px; border: 2px solid #E2012D;">
    <h2 style="color: #E2012D; margin: 0;">T1 ESPORTS</h2>
    <p style="color: #ccc; margin: 5px 0;">2024世界冠军</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("导航菜单")
page = st.sidebar.radio("选择分析模块", [
    "🏠 项目概览与数据总览",
    "📊 T1整体表现分析",
    "👥 T1选手深度分析",
    "🔄 团队协同与节奏分析",
    "🎮 英雄池与BP分析",
    "⭐ 各位置顶尖选手对比",
    "📈 深度数据洞察"
])

# 数据预处理
t1_data = player_df[player_df['TeamName'] == 'T1']
other_teams = player_df[player_df['TeamName'] != 'T1']

# 项目概览页面
if page == "🏠 项目概览与数据总览":
    # 添加背景音乐（请确保文件路径正确）
    try:
        autoplay_audio("M5000012VkGk2koUfA.mp3")
    except:
        st.sidebar.info("如需背景音乐，请将 'legends_never_die.mp3' 文件放在项目目录中")

    st.markdown('<div class="section-header">🎯 项目背景与目标</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        # 项目目标 - 使用简单的markdown
        st.markdown("### 项目目标")
        st.markdown("""
        - **深度解析T1战队夺冠关键因素**：通过数据可视化揭示T1战队的制胜之道
        - **多维度选手表现评估**：超越传统KDA的全面选手能力评估体系
        - **战术模式识别**：通过数据发现获胜模式和团队协同特点
        - **专业电竞洞察**：为电竞爱好者和分析师提供数据驱动的深度洞察
        """)

        st.markdown("### 数据集概况")
        st.markdown("""
        - **选手数据**：56名参赛选手，25项关键性能指标
        - **英雄数据**：热门英雄的选取率、禁用率、胜率等数据
        - **数据完整性**：已清洗整理，可直接用于深度分析
        """)

    with col2:
        # T1选手基本信息卡片
        st.markdown("### 🏆 T1冠军阵容")

        for _, player in t1_data.iterrows():
            player_name = player['PlayerName'].lower()
            if player_name in PLAYER_INFO:
                info = PLAYER_INFO[player_name]

                # 使用简单的卡片布局，字体颜色改为白色
                with st.container():
                    st.markdown(f"""
                    <div class="t1-player-card">
                        <h4 style="color: white;">🎯 {info['name']} - {info['position']}</h4>
                        <p style="color: white;"><strong>{info['real_name']}</strong></p>
                        <p style="color: white;">{info['description'][:80]}...</p>
                        <div style="display: flex; justify-content: space-between; color: white;">
                            <span>📊 胜率: <b>{player['Win rate']:.1%}</b></span>
                            <span>⚔️ KDA: <b>{player['KDA']:.2f}</b></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # 选手详细介绍
    st.markdown('<div class="section-header">🌟 T1选手详细介绍</div>', unsafe_allow_html=True)

    for player_name, info in PLAYER_INFO.items():
        player_data = t1_data[t1_data['PlayerName'].str.lower() == player_name]
        if not player_data.empty:
            player_stats = player_data.iloc[0]

            st.markdown(f'<div class="subsection-header">{info["name"]} - {info["position"]}</div>',
                        unsafe_allow_html=True)

            col1, col2 = st.columns([1, 2])

            with col1:
                # 选手图片
                try:
                    st.image(info["image_url"], use_container_width=True, caption=info["real_name"])
                except:
                    # 使用占位图
                    st.markdown(f"""
                    <div style="width: 100%; height: 300px; background: linear-gradient(135deg, #1a0000 0%, #330000 100%); 
                         display: flex; align-items: center; justify-content: center; border-radius: 15px; border: 2px solid #E2012D;">
                        <div style="text-align: center;">
                            <div style="width: 120px; height: 120px; background: #E2012D; border-radius: 50%; 
                                 display: flex; align-items: center; justify-content: center; margin: 0 auto 20px;">
                                <span style="color: white; font-size: 2rem; font-weight: bold;">{info['name'][0]}</span>
                            </div>
                            <h3 style="color: #E2012D; margin: 0;">{info['name']}</h3>
                            <p style="color: #ccc; margin: 5px 0;">{info['position']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                # 选手信息和统计数据 - 使用简单的布局
                st.markdown(f"### {info['real_name']}")
                st.markdown(f"{info['description']}")

                # 使用列来显示统计数据
                col_a, col_b, col_c, col_d = st.columns(4)

                with col_a:
                    st.metric("🎯 胜率", f"{player_stats['Win rate']:.1%}")
                with col_b:
                    st.metric("⚔️ KDA", f"{player_stats['KDA']:.2f}")
                with col_c:
                    st.metric("💥 分均伤害", f"{player_stats['DPM']:.0f}")
                with col_d:
                    st.metric("🤝 参团率", f"{player_stats['KP%']:.1%}")

    # 关键指标概览
    st.markdown('<div class="section-header">📈 关键指标概览</div>', unsafe_allow_html=True)

    # 第一行指标
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_win_rate = t1_data['Win rate'].mean()
        other_avg_win_rate = other_teams['Win rate'].mean() if len(other_teams) > 0 else 0
        win_rate_diff = avg_win_rate - other_avg_win_rate
        st.metric("T1平均胜率", f"{avg_win_rate:.1%}", f"+{win_rate_diff:.1%}")

    with col2:
        avg_kda = t1_data['KDA'].mean()
        other_avg_kda = other_teams['KDA'].mean() if len(other_teams) > 0 else 0
        kda_diff = avg_kda - other_avg_kda
        st.metric("T1平均KDA", f"{avg_kda:.2f}", f"+{kda_diff:.2f}")

    with col3:
        total_games = t1_data['Games'].sum()
        st.metric("T1总比赛场次", f"{total_games}场")

    with col4:
        t1_players_count = len(t1_data)
        st.metric("T1选手数量", f"{t1_players_count}人")

    # 第二行指标
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        avg_dpm = t1_data['DPM'].mean()
        other_avg_dpm = other_teams['DPM'].mean() if len(other_teams) > 0 else 0
        dpm_diff = avg_dpm - other_avg_dpm
        st.metric("T1平均分均伤害", f"{avg_dpm:.0f}", f"+{dpm_diff:.0f}")

    with col6:
        avg_gold = t1_data['GoldPerMin'].mean()
        other_avg_gold = other_teams['GoldPerMin'].mean() if len(other_teams) > 0 else 0
        gold_diff = avg_gold - other_avg_gold
        st.metric("T1平均分均经济", f"{avg_gold:.0f}", f"+{gold_diff:.0f}")

    with col7:
        avg_kp = t1_data['KP%'].mean()
        other_avg_kp = other_teams['KP%'].mean() if len(other_teams) > 0 else 0
        kp_diff = avg_kp - other_avg_kp
        st.metric("T1平均参团率", f"{avg_kp:.1%}", f"+{kp_diff:.1%}")

    with col8:
        avg_gd15 = t1_data['GD@15'].mean()
        other_avg_gd15 = other_teams['GD@15'].mean() if len(other_teams) > 0 else 0
        gd15_diff = avg_gd15 - other_avg_gd15
        st.metric("T1平均15分钟经济差", f"{avg_gd15:.0f}", f"+{gd15_diff:.0f}")

    # 数据集预览
    st.markdown('<div class="section-header">📋 数据集预览</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["选手数据", "英雄数据"])

    with tab1:
        st.subheader("选手数据概览")
        st.dataframe(player_df.head(10), use_container_width=True)
        st.write(f"数据集形状: {player_df.shape}")

    with tab2:
        st.subheader("英雄数据概览")
        st.dataframe(champions_df.head(10), use_container_width=True)
        st.write(f"数据集形状: {champions_df.shape}")



# T1整体表现分析页面
elif page == "📊 T1整体表现分析":
    st.markdown('<div class="section-header">📊 T1 vs 其他战队整体表现对比</div>', unsafe_allow_html=True)

    # 选择对比指标
    col1, col2 = st.columns([1, 3])

    with col1:
        metrics = st.multiselect(
            "选择对比指标",
            ['Win rate', 'KDA', 'Avg kills', 'Avg deaths', 'Avg assists',
             'GoldPerMin', 'KP%', 'DPM', 'GD@15', 'XPD@15', 'CSD@15'],
            default=['Win rate', 'KDA', 'GoldPerMin', 'DPM', 'GD@15']
        )

    if metrics:
        # 计算平均值
        t1_avg = t1_data[metrics].mean()
        other_avg = other_teams[metrics].mean()

        comparison_df = pd.DataFrame({
            'T1': t1_avg,
            '其他战队平均': other_avg
        })

        # 创建对比柱状图
        fig = go.Figure()
        fig.add_trace(go.Bar(name='T1', x=comparison_df.index, y=comparison_df['T1'],
                             marker_color=T1_RED, marker_line_color='darkred', marker_line_width=1.5))
        fig.add_trace(go.Bar(name='其他战队平均', x=comparison_df.index, y=comparison_df['其他战队平均'],
                             marker_color='lightgray', marker_line_color='gray', marker_line_width=1))

        fig.update_layout(
            title='T1 vs 其他战队关键指标对比',
            xaxis_title='指标',
            yaxis_title='数值',
            barmode='group',
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # 经济领先分析
    st.markdown('<div class="subsection-header">💰 经济领先与资源控制分析</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # 经济指标对比
        economic_metrics = ['GD@15', 'CSD@15', 'XPD@15', 'GoldPerMin']
        t1_economic = t1_data[economic_metrics].mean()
        other_economic = other_teams[economic_metrics].mean()

        fig = go.Figure()
        fig.add_trace(go.Bar(name='T1', x=economic_metrics, y=t1_economic,
                             marker_color=T1_RED))
        fig.add_trace(go.Bar(name='其他战队', x=economic_metrics, y=other_economic,
                             marker_color='lightblue'))

        fig.update_layout(
            title='经济指标对比',
            xaxis_title='经济指标',
            yaxis_title='数值',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 经济差分布
        fig = px.box(player_df, x='TeamName', y='GD@15',
                     title='各战队15分钟经济差分布')
        fig.update_layout(
            xaxis_title='战队',
            yaxis_title='15分钟经济差(GD@15)',
            height=400
        )
        # 高亮T1
        fig.add_hline(y=t1_data['GD@15'].mean(), line_dash="dash", line_color=T1_RED,
                      annotation_text="T1平均")
        st.plotly_chart(fig, use_container_width=True)

    # 伤害与输出分析
    st.markdown('<div class="subsection-header">⚔️ 伤害输出与效率分析</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        # 伤害相关指标
        damage_metrics = ['DPM', 'DamagePercent', 'Avg kills']
        t1_damage = t1_data[damage_metrics].mean()
        other_damage = other_teams[damage_metrics].mean()

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=t1_damage.values,
            theta=damage_metrics,
            fill='toself',
            name='T1',
            line_color=T1_RED
        ))
        fig.add_trace(go.Scatterpolar(
            r=other_damage.values,
            theta=damage_metrics,
            fill='toself',
            name='其他战队',
            line_color='lightblue'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True)
            ),
            title='伤害输出能力雷达图',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # DPM vs KDA 散点图
        fig = px.scatter(player_df, x='DPM', y='KDA', color='TeamName',
                         size='GoldPerMin', hover_name='PlayerName',
                         title='选手伤害输出vs生存能力 (DPM vs KDA)',
                         color_discrete_map={'T1': T1_RED})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# T1选手深度分析页面
elif page == "👥 T1选手深度分析":
    st.markdown('<div class="section-header">👥 T1选手个人表现深度分析</div>', unsafe_allow_html=True)

    # 选择选手
    t1_players = t1_data['PlayerName'].unique()
    selected_player = st.selectbox("选择T1选手", t1_players)

    player_data = player_df[player_df['PlayerName'] == selected_player].iloc[0]
    position_data = player_df[player_df['Position'] == player_data['Position']]

    # 选手基本信息卡片
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎯 胜率", f"{player_data['Win rate']:.1%}",
                  f"同位置排名: {len(position_data[position_data['Win rate'] > player_data['Win rate']]) + 1}/{len(position_data)}")
    with col2:
        st.metric("⚔️ KDA", f"{player_data['KDA']:.2f}",
                  f"同位置排名: {len(position_data[position_data['KDA'] > player_data['KDA']]) + 1}/{len(position_data)}")
    with col3:
        st.metric("💥 分均伤害", f"{player_data['DPM']:.0f}",
                  f"同位置排名: {len(position_data[position_data['DPM'] > player_data['DPM']]) + 1}/{len(position_data)}")
    with col4:
        st.metric("🤝 参团率", f"{player_data['KP%']:.1%}",
                  f"同位置排名: {len(position_data[position_data['KP%'] > player_data['KP%']]) + 1}/{len(position_data)}")

    # 选手与同位置对比
    st.markdown(f'<div class="subsection-header">📊 {selected_player} vs 同位置选手对比</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # KDA分布对比
        fig = px.box(position_data, y='KDA', title=f'{player_data["Position"]}位置KDA分布')
        fig.add_hline(y=player_data['KDA'], line_dash="dash", line_color=T1_RED,
                      annotation_text=f"{selected_player}: {player_data['KDA']:.2f}")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 核心指标对比雷达图
        comparison_metrics = ['KDA', 'DPM', 'GoldPerMin', 'KP%', 'GD@15']
        player_values = [player_data[metric] for metric in comparison_metrics]
        position_avg = [position_data[metric].mean() for metric in comparison_metrics]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=player_values,
            theta=comparison_metrics,
            fill='toself',
            name=selected_player,
            line_color=T1_RED
        ))
        fig.add_trace(go.Scatterpolar(
            r=position_avg,
            theta=comparison_metrics,
            fill='toself',
            name='同位置平均',
            line_color='lightblue'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True)
            ),
            title=f'{selected_player} vs 同位置平均核心指标',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # 选手专属分析
    if selected_player == "faker":
        st.markdown('<div class="subsection-header">👑 Faker - 中单之神深度分析</div>', unsafe_allow_html=True)

        mid_players = player_df[player_df['Position'] == 'Mid']

        col1, col2 = st.columns(2)

        with col1:
            # 中单选手DPM vs KDA
            fig = px.scatter(mid_players, x='DPM', y='KDA', size='GoldPerMin',
                             color='TeamName', hover_name='PlayerName',
                             title='中单选手伤害输出vs生存能力',
                             color_discrete_map={'T1': T1_RED})
            fig.add_annotation(x=player_data['DPM'], y=player_data['KDA'],
                               text="Faker", showarrow=True, arrowhead=2, bgcolor=T1_RED)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 对线能力分析
            laning_metrics = ['GD@15', 'CSD@15', 'XPD@15']
            faker_laning = player_data[laning_metrics]
            mid_avg_laning = mid_players[laning_metrics].mean()

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Faker', x=laning_metrics, y=faker_laning,
                                 marker_color=T1_RED))
            fig.add_trace(go.Bar(name='中单平均', x=laning_metrics, y=mid_avg_laning,
                                 marker_color='lightgray'))

            fig.update_layout(title='Faker对线期表现 vs 中单平均')
            st.plotly_chart(fig, use_container_width=True)

    elif selected_player == "zeus":
        st.markdown('<div class="subsection-header">⚡ Zeus - 对线压制力分析</div>', unsafe_allow_html=True)

        top_players = player_df[player_df['Position'] == 'Top']

        col1, col2 = st.columns(2)

        with col1:
            # 对线优势散点图
            fig = px.scatter(top_players, x='GD@15', y='Solo Kills', size='KDA',
                             color='TeamName', hover_name='PlayerName',
                             title='上单选手对线期优势分析',
                             color_discrete_map={'T1': T1_RED})
            fig.add_annotation(x=player_data['GD@15'], y=player_data['Solo Kills'],
                               text="Zeus", showarrow=True, arrowhead=2, bgcolor=T1_RED)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Zeus各项指标对比
            zeus_metrics = ['KDA', 'DPM', 'GD@15', 'Solo Kills', 'KP%']
            zeus_values = [player_data[metric] for metric in zeus_metrics]
            top_avg_values = [top_players[metric].mean() for metric in zeus_metrics]

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Zeus', x=zeus_metrics, y=zeus_values,
                                 marker_color=T1_RED))
            fig.add_trace(go.Bar(name='上单平均', x=zeus_metrics, y=top_avg_values,
                                 marker_color='lightgray'))

            fig.update_layout(title='Zeus核心指标 vs 上单平均')
            st.plotly_chart(fig, use_container_width=True)

# 团队协同与节奏分析页面
elif page == "🔄 团队协同与节奏分析":
    st.markdown('<div class="section-header">🔄 T1团队协同与比赛节奏分析</div>', unsafe_allow_html=True)

    # 视野控制分析
    st.markdown('<div class="subsection-header">👁️ 视野控制能力分析</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # 视野指标对比
        vision_metrics = ['Avg WPM', 'Avg WCPM', 'Avg VWPM', 'VSPM']
        t1_vision = t1_data[vision_metrics].mean()
        other_vision = other_teams[vision_metrics].mean()

        fig = go.Figure()
        fig.add_trace(go.Bar(name='T1', x=vision_metrics, y=t1_vision,
                             marker_color=T1_RED))
        fig.add_trace(go.Bar(name='其他战队', x=vision_metrics, y=other_vision,
                             marker_color='lightblue'))

        fig.update_layout(
            title='视野控制指标对比',
            xaxis_title='视野指标',
            yaxis_title='数值',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # T1选手视野控制热力图
        vision_data = t1_data[['PlayerName', 'Avg WPM', 'Avg WCPM', 'Avg VWPM', 'VSPM']].set_index('PlayerName')
        fig = px.imshow(vision_data,
                        title='T1选手视野控制指标热力图',
                        color_continuous_scale='Reds',
                        aspect='auto')
        st.plotly_chart(fig, use_container_width=True)

    # 节奏控制分析
    st.markdown('<div class="subsection-header">⏱️ 比赛节奏控制分析</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        # 前期节奏指标
        early_metrics = ['FB %', 'GD@15', 'XPD@15']
        t1_early = t1_data[early_metrics].mean()
        other_early = other_teams[early_metrics].mean()

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=t1_early.values,
            theta=early_metrics,
            fill='toself',
            name='T1',
            line_color=T1_RED
        ))
        fig.add_trace(go.Scatterpolar(
            r=other_early.values,
            theta=early_metrics,
            fill='toself',
            name='其他战队',
            line_color='lightblue'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True)
            ),
            title='前期节奏控制能力'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # 团队协同指标
        synergy_metrics = ['KP%', 'Avg assists', 'GoldPerMin']
        t1_synergy = t1_data[synergy_metrics].mean()
        other_synergy = other_teams[synergy_metrics].mean()

        fig = go.Figure()
        fig.add_trace(go.Bar(name='T1', x=synergy_metrics, y=t1_synergy,
                             marker_color=T1_RED))
        fig.add_trace(go.Bar(name='其他战队', x=synergy_metrics, y=other_synergy,
                             marker_color='lightblue'))

        fig.update_layout(
            title='团队协同指标对比',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

    # 击杀参与网络图（简化版）
    st.markdown('<div class="subsection-header">🔗 团队配合关系分析</div>', unsafe_allow_html=True)

    # 创建简化的网络图数据
    nodes = [{'name': player, 'group': 1} for player in t1_data['PlayerName']]

    # 基于KP%创建连接关系
    fig = go.Figure()

    # 添加节点
    for i, player in enumerate(t1_data['PlayerName']):
        kp = t1_data[t1_data['PlayerName'] == player]['KP%'].iloc[0]
        fig.add_trace(go.Scatter(
            x=[i], y=[kp * 10],  # 简化坐标
            mode='markers+text',
            marker=dict(size=50, color=T1_RED),
            text=player,
            textposition="middle center",
            name=player
        ))

    fig.update_layout(
        title='T1选手参团率分布（节点大小表示参团率）',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        showlegend=False,
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

# 英雄池与BP分析页面
elif page == "🎮 英雄池与BP分析":
    st.markdown('<div class="section-header">🎮 英雄选择与BP策略分析</div>', unsafe_allow_html=True)

    # 热门英雄分析
    st.markdown('<div class="subsection-header">🔥 赛事热门英雄分析</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # 选取率最高的英雄
        top_pick_champs = champions_df.nlargest(10, 'Picks')
        fig = px.bar(top_pick_champs, x='Picks', y='Champion',
                     title='选取率最高的10个英雄',
                     color='Winrate', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 禁用率最高的英雄
        top_ban_champs = champions_df.nlargest(10, 'Bans')
        fig = px.bar(top_ban_champs, x='Bans', y='Champion',
                     title='禁用率最高的10个英雄',
                     color='Presence', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    # 英雄表现分析
    st.markdown('<div class="subsection-header">📊 英雄表现深度分析</div>', unsafe_allow_html=True)


    # 数据预处理函数
    def clean_champion_data(df):
        """清洗英雄数据，处理百分比和时间格式"""
        cleaned_df = df.copy()

        # 处理百分比列
        percentage_cols = ['Presence', 'Winrate']
        for col in percentage_cols:
            if col in cleaned_df.columns:
                # 移除百分比符号并转换为浮点数
                cleaned_df[col] = cleaned_df[col].astype(str).str.replace('%', '')
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce') / 100

        # 处理时间列（GT - 游戏时间）
        if 'GT' in cleaned_df.columns:
            # 将时间格式 "HH:MM:SS" 转换为分钟数
            def time_to_minutes(time_str):
                if pd.isna(time_str) or time_str == '':
                    return np.nan
                try:
                    parts = str(time_str).split(':')
                    if len(parts) == 3:
                        hours, minutes, seconds = parts
                        return int(hours) * 60 + int(minutes) + int(seconds) / 60
                    return np.nan
                except:
                    return np.nan

            cleaned_df['GT_minutes'] = cleaned_df['GT'].apply(time_to_minutes)

        # 处理其他数值列
        numeric_cols = ['Picks', 'Bans', 'Wins', 'Losses', 'KDA', 'Avg BT', 'CSM', 'DPM', 'GPM', 'CSD@15', 'GD@15',
                        'XPD@15']
        for col in numeric_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

        return cleaned_df


    # 清洗数据
    cleaned_champions_df = clean_champion_data(champions_df)

    col3, col4 = st.columns(2)

    with col3:
        # 胜率 vs 选取率散点图 - 使用清洗后的数据
        valid_data = cleaned_champions_df.dropna(subset=['Picks', 'Winrate', 'KDA', 'Presence'])

        if not valid_data.empty:
            fig = px.scatter(valid_data, x='Picks', y='Winrate',
                             size='KDA', color='Presence',
                             hover_name='Champion',
                             hover_data=['Wins', 'Losses', 'KDA'],
                             title='英雄胜率 vs 选取率',
                             color_continuous_scale='Viridis')

            # 添加平均线
            avg_picks = valid_data['Picks'].mean()
            avg_winrate = valid_data['Winrate'].mean()
            fig.add_hline(y=avg_winrate, line_dash="dash", line_color="red",
                          annotation_text=f"平均胜率: {avg_winrate:.1%}")
            fig.add_vline(x=avg_picks, line_dash="dash", line_color="red",
                          annotation_text=f"平均选取: {avg_picks:.0f}")

            fig.update_layout(
                xaxis_title='选取次数',
                yaxis_title='胜率',
                yaxis_tickformat='.0%'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("没有足够的数据绘制胜率 vs 选取率散点图")

    with col4:
        # 英雄KDA分布 - 使用清洗后的数据
        kda_data = cleaned_champions_df.dropna(subset=['KDA'])

        if not kda_data.empty:
            # 计算KDA统计
            avg_kda = kda_data['KDA'].mean()
            max_kda = kda_data['KDA'].max()
            min_kda = kda_data['KDA'].min()

            fig = px.box(kda_data, y='KDA', title='英雄KDA分布')
            fig.add_hline(y=avg_kda, line_dash="dash", line_color="red",
                          annotation_text=f"平均KDA: {avg_kda:.2f}")

            fig.update_layout(
                yaxis_title='KDA',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # 显示KDA统计信息
            col4_1, col4_2, col4_3 = st.columns(3)
            with col4_1:
                st.metric("平均KDA", f"{avg_kda:.2f}")
            with col4_2:
                st.metric("最高KDA", f"{max_kda:.2f}")
            with col4_3:
                st.metric("最低KDA", f"{min_kda:.2f}")
        else:
            st.info("没有足够的数据绘制KDA分布图")

    # 英雄经济效率分析
    st.markdown('<div class="subsection-header">💰 英雄经济效率分析</div>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)

    with col5:
        # 伤害经济效率分析
        damage_efficiency_data = cleaned_champions_df.dropna(subset=['DPM', 'GPM'])

        if not damage_efficiency_data.empty:
            # 计算伤害经济比
            damage_efficiency_data = damage_efficiency_data.copy()
            damage_efficiency_data['Damage_per_Gold'] = damage_efficiency_data['DPM'] / damage_efficiency_data['GPM']

            # 找出效率最高的英雄
            top_efficiency = damage_efficiency_data.nlargest(5, 'Damage_per_Gold')

            fig = px.scatter(damage_efficiency_data, x='GPM', y='DPM',
                             size='Picks', color='Damage_per_Gold',
                             hover_name='Champion',
                             hover_data=['Winrate', 'KDA'],
                             title='英雄经济效率 vs 伤害输出',
                             color_continuous_scale='RdYlGn')

            # 标记效率最高的英雄
            for _, hero in top_efficiency.iterrows():
                fig.add_annotation(x=hero['GPM'], y=hero['DPM'],
                                   text=hero['Champion'], showarrow=True, arrowhead=1)

            fig.update_layout(
                xaxis_title='分均经济 (GPM)',
                yaxis_title='分均伤害 (DPM)'
            )
            st.plotly_chart(fig, use_container_width=True)

            # 显示效率最高的英雄
            st.subheader("💰 伤害经济效率最高的英雄")
            efficiency_display = top_efficiency[['Champion', 'Damage_per_Gold', 'DPM', 'GPM', 'Winrate']].round(3)
            st.dataframe(efficiency_display, use_container_width=True)
        else:
            st.info("没有足够的数据绘制经济效率散点图")

    with col6:
        # 英雄分类分析 - 基于选取率和胜率
        classification_data = cleaned_champions_df.dropna(subset=['Picks', 'Winrate', 'Presence'])

        if not classification_data.empty:
            # 创建英雄分类
            def classify_hero(row):
                picks = row['Picks']
                winrate = row['Winrate']

                if picks >= 20 and winrate >= 0.55:
                    return '热门强势'
                elif picks >= 20 and winrate < 0.45:
                    return '热门弱势'
                elif picks >= 20:
                    return '热门均衡'
                elif picks >= 10 and winrate >= 0.55:
                    return '潜力强势'
                elif picks < 10 and winrate >= 0.6:
                    return '冷门绝活'
                else:
                    return '一般英雄'


            classification_data = classification_data.copy()
            classification_data['Category'] = classification_data.apply(classify_hero, axis=1)

            # 按分类着色
            category_colors = {
                '热门强势': '#FF6B6B',
                '热门弱势': '#4ECDC4',
                '热门均衡': '#45B7D1',
                '潜力强势': '#96CEB4',
                '冷门绝活': '#FFEAA7',
                '一般英雄': '#DDA0DD'
            }

            fig = px.scatter(classification_data, x='Picks', y='Winrate',
                             color='Category', size='Presence',
                             hover_name='Champion',
                             hover_data=['KDA', 'Bans'],
                             title='英雄分类分析 (基于选取率和胜率)',
                             color_discrete_map=category_colors)

            # 添加分类区域线
            fig.add_hline(y=0.55, line_dash="dot", line_color="green")
            fig.add_hline(y=0.45, line_dash="dot", line_color="red")
            fig.add_vline(x=20, line_dash="dot", line_color="blue")
            fig.add_vline(x=10, line_dash="dot", line_color="orange")

            fig.update_layout(
                xaxis_title='选取次数',
                yaxis_title='胜率',
                yaxis_tickformat='.0%'
            )
            st.plotly_chart(fig, use_container_width=True)

            # 显示分类说明
            with st.expander("📋 英雄分类说明"):
                st.markdown("""
                - **热门强势**: 高选取率(≥20) + 高胜率(≥55%)
                - **热门弱势**: 高选取率(≥20) + 低胜率(<45%)  
                - **热门均衡**: 高选取率(≥20) + 中等胜率
                - **潜力强势**: 中等选取率(≥10) + 高胜率(≥55%)
                - **冷门绝活**: 低选取率(<10) + 极高胜率(≥60%)
                - **一般英雄**: 其他情况
                """)
        else:
            st.info("没有足够的数据绘制英雄分类分析图")

    # 新增：禁用率分析
    st.markdown('<div class="subsection-header">🚫 英雄禁用分析</div>', unsafe_allow_html=True)

    col7, col8 = st.columns(2)

    with col7:
        # 禁用率最高的英雄
        ban_data = cleaned_champions_df.dropna(subset=['Bans', 'Presence'])
        top_bans = ban_data.nlargest(10, 'Bans')

        if not top_bans.empty:
            fig = px.bar(top_bans, x='Bans', y='Champion',
                         color='Presence',
                         title='禁用率最高的10个英雄',
                         color_continuous_scale='Reds')

            fig.update_layout(
                xaxis_title='禁用次数',
                yaxis_title='英雄'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("没有足够的数据绘制禁用率图表")

    with col8:
        # 存在感分析（选取+禁用）
        presence_data = cleaned_champions_df.dropna(subset=['Presence', 'Winrate'])
        top_presence = presence_data.nlargest(10, 'Presence')

        if not top_presence.empty:
            fig = px.scatter(top_presence, x='Presence', y='Winrate',
                             size='Picks', color='Bans',
                             hover_name='Champion',
                             title='英雄存在感 vs 胜率 (前10名)',
                             color_continuous_scale='Blues')

            fig.update_layout(
                xaxis_title='存在感',
                yaxis_title='胜率',
                xaxis_tickformat='.0%',
                yaxis_tickformat='.0%'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("没有足够的数据绘制存在感图表")
# 各位置顶尖选手对比页面
elif page == "⭐ 各位置顶尖选手对比":
    st.markdown('<div class="section-header">⭐ 各位置顶尖选手表现对比</div>', unsafe_allow_html=True)

    position = st.selectbox("选择位置", ['Top', 'Jungle', 'Mid', 'Adc', 'Support'])

    # 根据位置选择关键指标
    position_metrics = {
        'Top': ['KDA', 'DPM', 'GD@15', 'Solo Kills', 'KP%'],
        'Jungle': ['KDA', 'KP%', 'VSPM', 'FB %', 'Avg assists'],
        'Mid': ['KDA', 'DPM', 'CSPerMin', 'GD@15', 'DamagePercent'],
        'Adc': ['KDA', 'DPM', 'GoldPerMin', 'DamagePercent', 'CSD@15'],
        'Support': ['KDA', 'KP%', 'Avg WPM', 'Avg WCPM', 'Avg assists']
    }

    position_data = player_df[player_df['Position'] == position]

    # 选择Top 5选手（按KDA）
    top_players = position_data.nlargest(5, 'KDA').copy()

    # 平行坐标图
    metrics = position_metrics[position]

    # 确保所有指标都有数据
    available_metrics = [metric for metric in metrics if metric in top_players.columns]

    if available_metrics and len(top_players) > 0:
        fig = px.parallel_coordinates(top_players,
                                      dimensions=available_metrics,
                                      color='KDA',
                                      color_continuous_scale='RdYlBu_r',
                                      title=f'{position}位置顶尖选手多维度对比')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"没有足够的数据绘制{position}位置的平行坐标图")

    # 详细对比表格
    st.markdown('<div class="subsection-header">📋 顶尖选手详细数据对比</div>', unsafe_allow_html=True)

    # 修复：避免重复列名
    base_cols = ['PlayerName', 'TeamName', 'Win rate']
    # 从metrics中移除已经在base_cols中的列
    unique_metrics = [metric for metric in metrics if metric not in base_cols]
    comparison_cols = base_cols + unique_metrics

    # 只选择存在的列
    existing_cols = [col for col in comparison_cols if col in top_players.columns]

    if len(top_players) > 0:
        display_data = top_players[existing_cols].reset_index(drop=True)

        # 确保没有重复列名
        if len(display_data.columns) != len(set(display_data.columns)):
            st.error("发现重复列名，正在修复...")
            # 如果有重复列名，使用唯一列名
            display_data = display_data.loc[:, ~display_data.columns.duplicated()]

        # 创建显示用的数据副本
        display_data_formatted = display_data.copy()

        # 格式化数值列
        for col in display_data_formatted.columns:
            if col not in ['PlayerName', 'TeamName']:
                # 根据列类型格式化
                if 'rate' in col.lower() or '%' in col or col == 'Win rate':
                    display_data_formatted[col] = display_data_formatted[col].apply(
                        lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A"
                    )
                elif isinstance(display_data_formatted[col].iloc[0], (int, float)):
                    display_data_formatted[col] = display_data_formatted[col].apply(
                        lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
                    )

        # 在选手名前添加T1标记
        display_data_formatted['PlayerName'] = display_data_formatted.apply(
            lambda row: f"🏆 {row['PlayerName']}" if row['TeamName'] == 'T1' else row['PlayerName'],
            axis=1
        )

        # 显示表格标题和说明
        st.markdown("**T1选手用🏆标记**")

        # 显示表格
        st.dataframe(display_data_formatted, use_container_width=True)

        # 备选方案：使用plotly表格
        with st.expander("📊 查看详细数据表格"):
            fig_table = go.Figure(data=[go.Table(
                header=dict(
                    values=list(display_data_formatted.columns),
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[display_data_formatted[col] for col in display_data_formatted.columns],
                    fill_color='lavender',
                    align='left'
                )
            )])
            fig_table.update_layout(
                title=f"{position}位置顶尖选手数据对比",
                height=400
            )
            st.plotly_chart(fig_table, use_container_width=True)

    else:
        st.info(f"没有找到{position}位置的选手数据")

    # 位置专属分析
    if position == 'Mid' and len(top_players) > 0:
        st.markdown('<div class="subsection-header">👑 中单选手深度对比</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # 中单伤害对比
            fig = px.bar(top_players, x='PlayerName', y='DPM',
                         title='顶尖中单分均伤害对比',
                         color='TeamName', color_discrete_map={'T1': T1_RED})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 中单经济效率
            fig = px.scatter(top_players, x='GoldPerMin', y='DPM',
                             size='KDA', color='TeamName',
                             hover_name='PlayerName',
                             title='中单经济效率 vs 伤害输出',
                             color_discrete_map={'T1': T1_RED})
            st.plotly_chart(fig, use_container_width=True)

    elif position == 'Top' and len(top_players) > 0:
        st.markdown('<div class="subsection-header">⚔️ 上单选手深度对比</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # 上单对线优势对比
            fig = px.bar(top_players, x='PlayerName', y='GD@15',
                         title='顶尖上单15分钟经济差对比',
                         color='TeamName', color_discrete_map={'T1': T1_RED})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 上单单杀能力
            fig = px.bar(top_players, x='PlayerName', y='Solo Kills',
                         title='顶尖上单单杀次数对比',
                         color='TeamName', color_discrete_map={'T1': T1_RED})
            st.plotly_chart(fig, use_container_width=True)

    elif position == 'Jungle' and len(top_players) > 0:
        st.markdown('<div class="subsection-header">🌲 打野选手深度对比</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # 打野视野控制
            fig = px.bar(top_players, x='PlayerName', y='VSPM',
                         title='顶尖打野视野分数对比',
                         color='TeamName', color_discrete_map={'T1': T1_RED})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 打野参团率
            fig = px.bar(top_players, x='PlayerName', y='KP%',
                         title='顶尖打野参团率对比',
                         color='TeamName', color_discrete_map={'T1': T1_RED})
            st.plotly_chart(fig, use_container_width=True)

    elif position == 'Adc' and len(top_players) > 0:
        st.markdown('<div class="subsection-header">🎯 ADC选手深度对比</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # ADC伤害占比
            fig = px.bar(top_players, x='PlayerName', y='DamagePercent',
                         title='顶尖ADC伤害占比对比',
                         color='TeamName', color_discrete_map={'T1': T1_RED})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # ADC经济转化
            fig = px.scatter(top_players, x='GoldPerMin', y='DPM',
                             size='KDA', color='TeamName',
                             hover_name='PlayerName',
                             title='ADC经济转化效率',
                             color_discrete_map={'T1': T1_RED})
            st.plotly_chart(fig, use_container_width=True)

    elif position == 'Support' and len(top_players) > 0:
        st.markdown('<div class="subsection-header">🛡️ 辅助选手深度对比</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # 辅助视野控制
            fig = px.bar(top_players, x='PlayerName', y='Avg WPM',
                         title='顶尖辅助分均插眼数对比',
                         color='TeamName', color_discrete_map={'T1': T1_RED})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 辅助参团率
            fig = px.bar(top_players, x='PlayerName', y='KP%',
                         title='顶尖辅助参团率对比',
                         color='TeamName', color_discrete_map={'T1': T1_RED})
            st.plotly_chart(fig, use_container_width=True)

# 深度数据洞察页面
elif page == "📈 深度数据洞察":
    st.markdown('<div class="section-header">📈 深度数据洞察与模式识别</div>', unsafe_allow_html=True)

    # 获胜因素相关性分析
    st.markdown('<div class="subsection-header">🔗 获胜关键因素相关性分析</div>', unsafe_allow_html=True)

    # 选择数值型列进行相关性分析
    numeric_cols = player_df.select_dtypes(include=[np.number]).columns
    correlation_data = player_df[numeric_cols].corr()

    # 重点关注与胜率的相关性
    win_rate_corr = correlation_data['Win rate'].sort_values(ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        # 显示与胜率最相关的指标
        top_correlations = win_rate_corr[1:11]  # 排除胜率自身
        fig = px.bar(x=top_correlations.values, y=top_correlations.index,
                     orientation='h', title='与胜率最相关的指标（正相关）',
                     color=top_correlations.values,
                     color_continuous_scale='Reds')
        fig.update_layout(yaxis_title='指标', xaxis_title='相关系数')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 显示与胜率最负相关的指标
        bottom_correlations = win_rate_corr[-10:]
        fig = px.bar(x=bottom_correlations.values, y=bottom_correlations.index,
                     orientation='h', title='与胜率最负相关的指标',
                     color=bottom_correlations.values,
                     color_continuous_scale='Blues_r')
        fig.update_layout(yaxis_title='指标', xaxis_title='相关系数')
        st.plotly_chart(fig, use_container_width=True)

    # 聚类分析
    st.markdown('<div class="subsection-header">🎯 选手表现聚类分析</div>', unsafe_allow_html=True)

    try:
        # 使用简化的聚类方法避免threadpoolctl问题
        features_for_clustering = ['KDA', 'DPM', 'GoldPerMin', 'KP%', 'Win rate', 'GD@15']

        # 确保所有特征都存在
        available_features = [feature for feature in features_for_clustering if feature in player_df.columns]

        if len(available_features) < 2:
            st.warning("可用于聚类的特征不足")
        else:
            # 创建数据副本
            clustering_data = player_df[available_features + ['PlayerName', 'TeamName', 'Position']].copy()

            # 处理数值列
            for col in available_features:
                clustering_data[col] = pd.to_numeric(clustering_data[col], errors='coerce')

            clustering_data = clustering_data.dropna()

            if len(clustering_data) < 4:
                st.warning("有效数据样本不足，无法进行聚类分析")
            else:
                # 使用简化的手动聚类方法
                st.info("使用简化聚类方法分析选手表现...")

                # 基于KDA和DPM进行简单分组
                kda_mean = clustering_data['KDA'].mean()
                dpm_mean = clustering_data['DPM'].mean()


                def simple_cluster(row):
                    kda = row['KDA']
                    dpm = row['DPM']

                    if kda > kda_mean and dpm > dpm_mean:
                        return '顶尖选手'
                    elif kda > kda_mean and dpm <= dpm_mean:
                        return '生存专家'
                    elif kda <= kda_mean and dpm > dpm_mean:
                        return '输出机器'
                    else:
                        return '一般选手'


                clustering_data['Cluster_Group'] = clustering_data.apply(simple_cluster, axis=1)

                # 定义颜色映射
                cluster_colors = {
                    '顶尖选手': T1_RED,
                    '生存专家': 'green',
                    '输出机器': 'blue',
                    '一般选手': 'purple'
                }

                # 可视化聚类结果
                col3, col4 = st.columns(2)

                with col3:
                    fig = px.scatter(
                        clustering_data,
                        x='KDA',
                        y='DPM',
                        color='Cluster_Group',
                        color_discrete_map=cluster_colors,
                        hover_data=['PlayerName', 'TeamName', 'Position'],
                        title='选手表现分组分析 (KDA vs DPM)',
                        labels={'Cluster_Group': '选手类型'}
                    )

                    # 添加平均值线
                    fig.add_hline(y=dpm_mean, line_dash="dash", line_color="gray",
                                  annotation_text=f"平均DPM: {dpm_mean:.0f}")
                    fig.add_vline(x=kda_mean, line_dash="dash", line_color="gray",
                                  annotation_text=f"平均KDA: {kda_mean:.2f}")

                    # 高亮T1选手
                    t1_players = clustering_data[clustering_data['TeamName'] == 'T1']
                    if not t1_players.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=t1_players['KDA'],
                                y=t1_players['DPM'],
                                mode='markers',
                                marker=dict(
                                    size=12,
                                    color=T1_RED,
                                    symbol='star',
                                    line=dict(width=3, color='white')
                                ),
                                name='T1选手',
                                showlegend=True
                            )
                        )

                    st.plotly_chart(fig, use_container_width=True)

                with col4:
                    # 显示分组统计
                    st.markdown("#### 📊 分组统计")
                    group_stats = clustering_data.groupby('Cluster_Group').agg({
                        'KDA': ['mean', 'count'],
                        'DPM': 'mean',
                        'Win rate': 'mean',
                        'GoldPerMin': 'mean'
                    }).round(2)

                    # 格式化统计表
                    stats_display = pd.DataFrame({
                        '选手数量': group_stats[('KDA', 'count')],
                        '平均KDA': group_stats[('KDA', 'mean')],
                        '平均DPM': group_stats[('DPM', 'mean')],
                        '平均胜率': group_stats[('Win rate', 'mean')],
                        '平均分均经济': group_stats[('GoldPerMin', 'mean')]
                    })

                    st.dataframe(stats_display, use_container_width=True)

                    # 显示T1选手分组情况
                    st.markdown("#### 🏆 T1选手分组")
                    t1_clusters = clustering_data[clustering_data['TeamName'] == 'T1'][['PlayerName', 'Cluster_Group']]
                    if not t1_clusters.empty:
                        st.dataframe(t1_clusters.reset_index(drop=True), use_container_width=True)
                    else:
                        st.info("聚类数据中未找到T1选手")

                # 分组描述
                st.markdown("""
                ### 🎯 分组解读
                - **顶尖选手**: KDA和伤害输出都高于平均水平的全能选手
                - **生存专家**: KDA高但伤害输出一般的稳健型选手  
                - **输出机器**: 伤害输出高但KDA一般的激进型选手
                - **一般选手**: KDA和伤害输出都低于平均水平的选手
                """)

    except Exception as e:
        st.error(f"分析过程中出现错误: {str(e)}")
        # 显示基础的可视化作为备选
        st.info("显示基础选手数据分析...")

        try:
            # 简单的KDA vs DPM散点图
            simple_data = player_df[['PlayerName', 'KDA', 'DPM', 'TeamName', 'Position']].dropna()
            if not simple_data.empty:
                fig = px.scatter(
                    simple_data,
                    x='KDA',
                    y='DPM',
                    color='TeamName',
                    hover_name='PlayerName',
                    hover_data=['Position'],
                    title='选手KDA vs DPM分布',
                    color_discrete_map={'T1': T1_RED}
                )
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("无法显示基础可视化")

    # T1夺冠关键因素总结
    st.markdown('<div class="subsection-header">🏆 T1夺冠关键因素总结</div>', unsafe_allow_html=True)

    # 计算实际的数据差异
    try:
        kp_diff = t1_data['KP%'].mean() - other_teams['KP%'].mean()
        dpm_diff = t1_data['DPM'].mean() - other_teams['DPM'].mean()
        gold_diff = t1_data['GoldPerMin'].mean() - other_teams['GoldPerMin'].mean()
        gd15_diff = t1_data['GD@15'].mean() - other_teams['GD@15'].mean()
        vspm_diff = t1_data['VSPM'].mean() - other_teams['VSPM'].mean()
        assists_diff = t1_data['Avg assists'].mean() - other_teams['Avg assists'].mean()

        col5, col6, col7 = st.columns(3)

        with col5:
            st.metric("🎯 团队协同", "卓越", f"参团率领先 +{kp_diff:.1%}")
            st.metric("💥 伤害输出", "顶尖", f"分均伤害领先 +{dpm_diff:.0f}")

        with col6:
            st.metric("💰 经济效率", "高效", f"分均经济领先 +{gold_diff:.0f}")
            st.metric("⏱️ 前期节奏", "压制", f"15分钟经济差领先 +{gd15_diff:.0f}")

        with col7:
            st.metric("👁️ 视野控制", "精密", f"视野分数领先 +{vspm_diff:.1f}")
            st.metric("🤝 选手配合", "默契", f"助攻数领先 +{assists_diff:.1f}")

    except Exception as e:
        # 显示默认值作为备选
        col5, col6, col7 = st.columns(3)

        with col5:
            st.metric("🎯 团队协同", "卓越", "参团率领先")
            st.metric("💥 伤害输出", "顶尖", "分均伤害领先")

        with col6:
            st.metric("💰 经济效率", "高效", "分均经济领先")
            st.metric("⏱️ 前期节奏", "压制", "前期经济领先")

        with col7:
            st.metric("👁️ 视野控制", "精密", "视野控制优秀")
            st.metric("🤝 选手配合", "默契", "团队配合出色")
# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><b>2024英雄联盟世界赛T1夺冠深度分析项目</b> | 使用Streamlit构建 | 数据驱动电竞分析</p>
    <p>数据来源: 2024 LOL Championship Player Stats & Champions Data</p>
</div>
""", unsafe_allow_html=True)