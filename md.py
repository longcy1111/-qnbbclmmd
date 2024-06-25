import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance
import os
from fastai.vision.all import *
import pathlib
import sys
import pandas as pd
import pickle
import numpy as np
from io import BytesIO

# 根据不同的操作系统设置正确的pathlib.Path
if sys.platform == "win32":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath

# 获取当前文件所在的文件夹路径
path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path,"face_shapes_model.pkl")

# 加载模型
learn_inf = load_learner(model_path)

# 恢复pathlib.Path的原始值
if sys.platform == "win32":
    pathlib.PosixPath = temp
else:
    pathlib.WindowsPath = temp

# HTML 内容与内联 CSS 样式
html_content = """
<div style="color: black; font-size: 50px; font-weight: bold; font-family: Arial; text-decoration: underline; background-color: pink; margin: 60px; border: 2px solid black; text-align: center;">
    绝无仅有的颜值打分器！</div>
*<span style='font-size: 15px;'>由严谨但不专业的团队一手打造♡(*´∀｀*)人(*´∀｀*)♡.</span>*
"""
# 使用 st.write() 来显示 HTML 内容
st.write(html_content, unsafe_allow_html=True)

st.header("瞧一瞧，看一看啦！")
st.write("动动您的小手测测吧╭☞( ￣ ▽￣)╭☞")
st.write("只需一张您的美照，我们将给出您的脸型预测~ξ( ✿＞◡❛)")

# 允许用户上传图片
uploaded_file = st.file_uploader("选择小仙女的美照吧~最好是正面照呢，结果会更准确",type=["jpg","jpeg","png"])

# 初始化评分列表
if 'ratings' not in st.session_state:
    st.session_state.ratings = []

# 获取当前文件所在的文件夹路径path = os.path.dirname(os.path.abspath(__file__)) model_path = os.path.join(path,"best_model.pkl")

with open('/mount/src/qnbbclmmd-/best_model.pkl', 'rb') as file:
    rfr_model = pickle.load(file)
 加载模型
learn_inf = load_learner(model_path)




# 定义计算颜色直方图的函数
def compute_color_histogram_pil(image, bins=32):
    # Convert the image to the RGB color space
    r, g, b = image.split()
    # Compute the color histogram
    hist_r = r.histogram()
    hist_g = g.histogram()
    hist_b = b.histogram()
    # Normalize the histograms
    hist_r = [x / sum(hist_r) for x in hist_r]
    hist_g = [x / sum(hist_g) for x in hist_g]
    hist_b = [x / sum(hist_b) for x in hist_b]
    # Combine the histograms
    hist_combined = np.concatenate((hist_r, hist_g, hist_b))
    return hist_combined

descriptions = {
    "1-20": "  你的颜值你的颜值似乎还在沉睡中，等待着某个神秘的时刻醒来。别担心，每个人都有自己的魅力时刻，只是时间早晚而已。或许，下一次你换个发型、穿上新衣，就能立刻惊艳四座！",
    "21-40": "  你的颜值就像一颗未经雕琢的宝石，虽然外表普通，但内在却蕴藏着无尽的光芒。只需稍加打磨，你的魅力就会如同烟花般绽放，让人目不暇接。所以，别灰心，你的颜值潜力无限！",
    "41-60": "  你的颜值就像家常便饭，虽然普通，但绝对是让人离不开的那种！它可能不是最华丽、最惊艳的，但却是最能让人感到舒适和自在的。你的面容给人一种亲切和温暖的感觉，就像是一杯热茶，在寒冷的冬日里带给人温暖和安慰。",
    "61-70": "  你的颜值给人一种亲切自然的感觉，就像春日的暖阳，温暖而舒适。在人群中，你总能以你的真诚和善良吸引他人的目光。继续保持这份真实和自然，你的魅力会自然而然地散发出来。",
    "71-80": "  你的颜值已经相当出色了！五官精致，气质独特，走在街头巷尾总能吸引不少目光。你散发着一种自然的自信和优雅，让你在人群中显得与众不同。继续保持这份独特魅力。",
    "81-100": "  你的颜值简直是完美无瑕！无论是面容、身材还是气质，你都展现出了无可挑剔的美感。你的出现总能引起一阵惊叹和赞叹，让人为你的美丽而倾倒。你的独特魅力让人无法抗拒，无论是哪个角度，你都能散发出迷人的光彩。你的面容和气质相得益彰，无论是在派对聚会上还是在日常生活中，你总能轻易成为众人瞩目的中心。你的自信和从容让你在任何场合都光彩照人，继续保持这份风采!算我求你了！！"
}

# 如果用户已上传图片
if uploaded_file is not None:
    # 显示上传的图片
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的图片", use_column_width=False)

    # 检查文件类型（扩展名）
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in ["png", "jpg", "jpeg"]:
        st.error("不支持的文件格式。请上传 PNG、JPG 或 JPEG 格式的图片。")
    else:

        # 计算新图像的颜色直方图
        new_image_histogram = compute_color_histogram_pil(image)

        # 归一化直方图
        new_image_histogram_normalized = new_image_histogram / np.linalg.norm(new_image_histogram)

        # 将直方图转换为适合模型输入的格式
        new_image_features = new_image_histogram_normalized.reshape(1, -1)

        # 使用处理后的输入数据进行预测
        predicted_attractiveness = rfr_model.predict(new_image_features)
        predicted_score = predicted_attractiveness[0]

        # 假设原始分数范围是0到5
        min_score = 0
        max_score = 5

        # 转换为百分制
        percentage_score = round((predicted_score - min_score) / (max_score - min_score) * 100+10, 2)
        
        if st.button("请为我打分"):
            # 显示预测结果
            st.write(f'您的颜值分数： <span style="font-size:60px;">{percentage_score:.2f}</span>', unsafe_allow_html=True)

            # 根据分数添加颜值分阶描述
            score_range = ""
            if percentage_score >= 1 and percentage_score <= 20:
               score_range = "1-20"
            elif percentage_score >= 21 and percentage_score <= 40:
                score_range = "21-40"
            elif percentage_score >= 41 and percentage_score <= 60:
                score_range = "41-60"
            elif percentage_score >= 61 and percentage_score <= 70:
                score_range = "61-70"
            elif percentage_score >= 71 and percentage_score <= 80:
                score_range = "71-80"
            elif percentage_score >= 81 and percentage_score <= 100:
                score_range = "81-100"

            descriptions = descriptions.get(score_range, "")
            st.write(f"\n{descriptions}")

    # 添加滤镜选项
    filter_option = st.selectbox("选择一个滤镜", ["原图", "模糊", "轮廓", "细节增强", "边缘增强", "提高清晰度"])

    # 应用滤镜
    if filter_option == "模糊":
        filtered_image = image.filter(ImageFilter.BLUR)
    elif filter_option == "轮廓":
        filtered_image = image.filter(ImageFilter.CONTOUR)
    elif filter_option == "细节增强":
        filtered_image = image.filter(ImageFilter.DETAIL)
    elif filter_option == "边缘增强":
        filtered_image = image.filter(ImageFilter.EDGE_ENHANCE)
    elif filter_option == "提高清晰度":
        enhancer = ImageEnhance.Sharpness(image)
        filtered_image = enhancer.enhance(2.0)  # 提高清晰度
    else:
        filtered_image = image

    # 显示滤镜后的图片
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="原图", use_column_width=True)
    with col2:
        st.image(filtered_image, caption="滤镜后的图片", use_column_width=True)

    # 使用模型进行预测
    pred_class, pred_idx, probs = learn_inf.predict(PILImage.create(filtered_image))

    # 替换预测结果中的脸型名称
    if pred_class == "oval-face":
        pred_class_display = "精致瓜子脸"
    elif pred_class == "round-face":
        pred_class_display = "圆润鹅蛋脸"
    elif pred_class == "square-face":
        pred_class_display = "大气方圆脸"
    else:
        pred_class_display = pred_class  # 如果还有其他脸型，保持原样

    # 显示预测结果
    st.write(f'小仙女您的脸型是ε(*･ω･)_/ﾟ:･☆ <span style="font-size:30px;">{pred_class_display}</span>', unsafe_allow_html=True)
    st.write(f"置信度: {probs[pred_idx]:.2f}")


    # 添加评分滑块
    attractiveness_satisfaction = st.slider("您对我们的颜值打分结果的满意度如何？",min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    app_rating = st.slider("请给我们的应用打个分吧！", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    face_satisfaction = st.slider("请问您对自己脸型的满意度是多少呢？", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

    # 添加按钮
    if st.button("提交评分"):
        # 记录评分
        st.session_state.ratings.append((percentage_score, pred_class_display, attractiveness_satisfaction, face_satisfaction, app_rating))
        st.write("评分已提交，感谢您的反馈！")

else:
    st.warning("请上传图片文件")

# 显示所有用户的评分
if st.button("查看所有用户的评分"):
    st.write(f"目前已经有 {len(st.session_state.ratings)} 个用户评过分")
    st.write("评分如下：")
    for idx, (percentage_score, pred_class_display, attractiveness_satisfaction, face_satisfaction, app_rating ) in enumerate(st.session_state.ratings):
        st.write(f"用户 {idx + 1}: 颜值打分： {percentage_score:.2f}  —— 脸型： {pred_class_display},   —— 打分结果满意度： {attractiveness_satisfaction}   —— 脸型满意度： {face_satisfaction}    —— 应用评分： {app_rating}")

# 下载所有记录为Excel表格
if st.button("下载所有记录为Excel表格"):
    df = pd.DataFrame(st.session_state.ratings, columns=["颜值打分", "脸型", "打分结果满意度","脸型满意度","应用评分"])    
    df.to_excel("user_ratings.xlsx", index=False)
    st.write("所有记录已保存为 user_ratings.xlsx")
