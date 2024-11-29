import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Thư viện để load mô hình đã lưu


# Load mô hình đã huấn luyện
model = joblib.load(r"alzheimers_model.pkl")
# Tiêu đề ứng dụng
st.title(
    "Alzheimer's Disease Prediction")
st.image("ahzimer_pic.jpg")
# Giới thiệu
st.write("""
Nhập các thông tin bên dưới để dự đoán khả năng mắc bệnh Alzheimer:
""")

# Form để nhập thông tin
age = st.number_input("Age", min_value=60, max_value=100, step=1, value=60)
gender = st.selectbox("Gender", ["Male", "Female"])
## Functional Assessment
st.markdown("# Đánh giá chức năng(Functional assessment)")
st.info("Đánh giá chức năng là một quá trình hợp tác liên tục, kết hợp việc quan sát, đặt câu hỏi có ý nghĩa, lắng nghe những câu chuyện từ gia đình và phân tích các kỹ năng cũng như hành vi của bệnh nhân trong các hoạt động và thói quen hàng ngày diễn ra một cách tự nhiên, ở nhiều tình huống và môi trường khác nhau.")
st.warning("Chỉ số này cần phải được thu thập ở bệnh viện! Vậy nên chỉ số ở đây là chỉ số đánh giá tự phát, với thang điểm từ 0 đến 10, cho biết mức độ FAS của một người")
fas = st.slider("Functional Assessment", min_value=0.0, max_value=10.0, step=0.2)
st.markdown("# Kiểm tra tráng thái tâm thần tối thiểu của bệnh nhân(MMSE)")
# Phần 1 của tính toán MMSE
st.markdown("## 1.Định hướng thời gian không gian và thời gian")

# Câu hỏi 1
st.write("Câu hỏi 1: Năm nay là năm nào?")
question_1_true = st.checkbox("Đúng", key="q1_true")
question_1_false = st.checkbox("Sai", key="q1_false")

# Kiểm tra chỉ có thể chọn một trong hai
if question_1_true and question_1_false:
    st.warning("Bạn chỉ có thể chọn một trong hai. Vui lòng bỏ chọn câu còn lại!")

# Câu hỏi 2
st.write("Câu hỏi 2: Mùa này là mùa nào?")
question_2_true = st.checkbox("Đúng", key="q2_true")
question_2_false = st.checkbox("Sai", key="q2_false")

if question_2_true and question_2_false:
    st.warning("Bạn chỉ có thể chọn một trong hai. Vui lòng bỏ chọn câu còn lại!")

# Câu hỏi 3
st.write("Câu hỏi 3: Tháng này là tháng nào?")
question_3_true = st.checkbox("Đúng", key="q3_true")
question_3_false = st.checkbox("Sai", key="q3_false")

if question_3_true and question_3_false:
    st.warning("Bạn chỉ có thể chọn một trong hai. Vui lòng bỏ chọn câu còn lại!")

# Câu hỏi 4
st.write("Câu hỏi 4: Hôm nay là ngày nào (Hoặc thứ mấy)?")
question_4_true = st.checkbox("Đúng", key="q4_true")
question_4_false = st.checkbox("Sai", key="q4_false")

if question_4_true and question_4_false:
    st.warning("Bạn chỉ có thể chọn một trong hai. Vui lòng bỏ chọn câu còn lại!")

# Câu hỏi 5
st.write("Câu hỏi 5: Ông/Bà đang ở thành phố/huyện nào?")
question_5_true = st.checkbox("Đúng", key="q5_true")
question_5_false = st.checkbox("Sai", key="q5_false")

if question_5_true and question_5_false:
    st.warning("Bạn chỉ có thể chọn một trong hai. Vui lòng bỏ chọn câu còn lại!")

# Câu hỏi 6
st.write("Câu hỏi 6: Ông/Bà đang ở bệnh viện nào?")
question_6_true = st.checkbox("Đúng", key="q6_true")
question_6_false = st.checkbox("Sai", key="q6_false")

if question_6_true and question_6_false:
    st.warning("Bạn chỉ có thể chọn một trong hai. Vui lòng bỏ chọn câu còn lại!")

# Câu hỏi 7
st.write("Câu hỏi 7: Ông/Bà đang ở khoa/tầng nào?")
question_7_true = st.checkbox("Đúng", key="q7_true")
question_7_false = st.checkbox("Sai", key="q7_false")

if question_7_true and question_7_false:
    st.warning("Bạn chỉ có thể chọn một trong hai. Vui lòng bỏ chọn câu còn lại!")

# Câu hỏi 8
st.write("Câu hỏi 8: Ông bà đang ở nước nào?")
question_8_true = st.checkbox("Đúng", key="q8_true")
question_8_false = st.checkbox("Sai", key="q8_false")

if question_8_true and question_8_false:
    st.warning("Bạn chỉ có thể chọn một trong hai. Vui lòng bỏ chọn câu còn lại!")
    
    
## Phần 2
st.markdown("## 2.Đăng kí ghi nhận")
st.info("Đọc tên 3 đồ vật 'cái bàn', 'đồng xu', 'quả táo' 1 cách chậm rãi, rõ ràng sau đó yêu cầu bệnh nhân nhắc lại")
st.write("Số câu trả lời đúng:")
question_part2_3=st.checkbox("Cả 3", key="part2_3") 
question_part2_2=st.checkbox("Hai", key="part2_2")
question_part2_1=st.checkbox("Một", key="part2_1")
question_part2_0=st.checkbox("Sai hết", key="part2_0")
if question_part2_3 and question_part2_2 or question_part2_1 or question_part2_0:
    st.warning("Bạn chỉ có thể chọn một trong bốn. Vui lòng bỏ chọn câu còn lại")
## Phần 3
st.markdown("## 3.Sự chú ý/Tính toán")
st.info("Yêu cầu bệnh nhân dùng phép tính lùi 100-7 liên tiếp (dừng lại sau5 lần). Hãy viết ra nháp các câu trả lời của bệnh nhân")
st.write("Số lần tính đúng:")
col1_3,col2_3=st.columns(2)
with col1_3:
    question_part3_5=st.checkbox("5", key="part3_5") 
    question_part3_4=st.checkbox("4", key="part3_4")
    question_part3_3=st.checkbox("3", key="part3_3")
with col2_3:
    question_part3_2=st.checkbox("2", key="part3_2")
    question_part3_1=st.checkbox("1", key="part3_1")
    question_part3_0=st.checkbox("0", key="part3_0")
selected_options=[question_part3_5,question_part3_4,question_part3_3,question_part3_2,question_part3_1,question_part3_0]
if sum(selected_options)>1 :
    st.warning("Bạn chỉ có thể chọn một trong sáu. Vui lòng bỏ chọn câu còn lại")
## Phần 4
st.markdown("## 4.Trí nhớ gần")
st.info("Yêu cầu bệnh nhân nhắc lại 3 đồ vật đã kể trước đó")
col1_4, col2_4=st.columns(2)
with col1_4:
    question_part4_3=st.checkbox("3", key="part4_3")
    question_part4_2=st.checkbox("2", key="part4_2")
with col2_4:
    question_part4_1=st.checkbox("1", key="part4_1")
    question_part4_0=st.checkbox("0", key="part4_0")
selected_options=[question_part4_3,question_part4_2,question_part4_1,question_part4_0]
if sum(selected_options)>1 :
    st.warning("Bạn chỉ có thể chọn một trong bốn. Vui lòng bỏ chọn câu còn lại")
# Phần 5
st.markdown("## 5.Ngôn ngữ")
st.markdown('### Đưa bệnh nhân cây bút và hỏi:"Đây là cái gì?"')
question5_1_true = st.checkbox("Đúng", key="q5_1_true")
question5_1_false = st.checkbox("Sai", key="q5_1_false")
if question5_1_true and question5_1_false :
    st.warning("Bạn chỉ có thể chọn một trong hai. Vui lòng bỏ chọn câu còn lại")

st.markdown('### Đưa bệnh nhân một chiếc đồng hồ và hỏi:"Đây là cái gì?"')
question5_2_true = st.checkbox("Đúng", key="q5_2_true")
question5_2_false = st.checkbox("Sai", key="q5_2_false")
if question5_2_true and question5_2_false :
    st.warning("Bạn chỉ có thể chọn một trong hai. Vui lòng bỏ chọn câu còn lại")
    
st.markdown('### Hãy nhắc lại câu này:"Không có nếu, và, hoặc nhưng"')
question5_3_true = st.checkbox("Đúng", key="q5_3_true")
question5_3_false = st.checkbox("Sai", key="q5_3_false")

if question5_3_true and question5_3_false :
    st.warning("Bạn chỉ có thể chọn một trong hai. Vui lòng bỏ chọn câu còn lại")
    
    
# Phần 6.Thực hiện mệnh lệnh phức tạp
st.markdown("## 6.Thực hiện mệnh lệnh phức tạp")
st.info('Đưa một mảnh giấy trắng và yêu cầu bệnh nhân bằng 1 câu "Cầm lấy tờ giầy bằng tay, gấp đôi tờ giấy lại và đặt nó xuống sàn"')
col1_6,col2_6=st.columns(2)
with col1_6:
    question_part6_0=st.checkbox("Không thực hiện được", key="part6_0")
    question_part6_1=st.checkbox("Làm đúng 1 bước", key="part6_1")
with col2_6:
    question_part6_2=st.checkbox("Làm đúng 2 bước", key="part6_2")
    question_part6_3=st.checkbox("Làm đúng 3 bước", key="part6_3")
selected_options = [question_part6_0, question_part6_1, question_part6_2, question_part6_3]
if sum(selected_options) > 1:  # Nếu có hơn một checkbox được chọn
    st.warning("Bạn chỉ có thể chọn một trong bốn. Vui lòng bỏ chọn các ô còn lại.")
st.info('Đưa bệnh nhân 1 tờ giấy có ghi rõ mệnh lệnh "HÃY NHẮM MẮT LẠI".Yêu cầu bệnh nhân đọc và làm theo')
question_part6_4=st.checkbox("Không làm đúng", key="part6_4")
question_part6_5=st.checkbox("Làm đúng", key="part6_5")
if question_part6_4 and question_part6_5:
    st.warning("Bạn chỉ có thể chọn một trong hai. Vui lòng bỏ chọn câu còn lại")
st.info('Đưa giấy bút yêu cầu bệnh nhân viết 1 câu bất kỳ:')
question_part6_6=st.checkbox("Viết được", key="part6_6")
question_part6_7=st.checkbox("Không viết được", key="part6_7")
if question_part6_6 and question_part6_7:
    st.warning("Bạn chỉ có thể chọn một trong hai. Vui lòng bỏ chọn câu còn lại")
st.info('Yêu cầu bệnh nhân vẽ lại hình sau. Vẽ được 2 hình 5 cạnh giao nhau, tạo nên hình tứ giác là vẽ đúng:')
question_part6_8=st.checkbox("Không vẽ đúng", key="part6_8")
question_part6_9=st.checkbox("Vẽ đúng", key="part6_9")
if question_part6_8 and question_part6_9:
    st.warning("Bạn chỉ có thể chọn một trong hai. Vui lòng bỏ chọn câu còn lại")
st.image("2_hinh_ngu_giac.png")


## Xử lý kết quả điểm MMSE
# Tính điểm MMSE
def calculate_score():
    score = 0
    # Phần 1
    if question_1_true and not question_1_false: score += 1
    if question_2_true and not question_2_false: score += 1
    if question_3_true and not question_3_false: score += 1
    if question_4_true and not question_4_false: score += 1
    if question_5_true and not question_5_false: score += 1
    if question_6_true and not question_6_false: score += 1
    if question_7_true and not question_7_false: score += 1
    if question_8_true and not question_8_false: score += 1

    # Phần 2
    if question_part2_1: score += 1
    if question_part2_2: score += 2
    if question_part2_3: score += 3

    # Phần 3
    if question_part3_1: score += 1
    if question_part3_2: score += 2
    if question_part3_3: score += 3
    if question_part3_4: score += 4
    if question_part3_5: score += 5

    # Phần 4
    if question_part4_1: score += 1
    if question_part4_2: score += 2
    if question_part4_3: score += 3

    # Phần 5
    if question5_1_true and not question5_1_false: score += 1
    if question5_2_true and not question5_2_false: score += 1
    if question5_3_true and not question5_3_false: score += 1

    # Phần 6
    if question_part6_1: score += 1
    if question_part6_2: score += 2
    if question_part6_3: score += 3
    if question_part6_5 and not question_part6_4: score += 1
    if question_part6_6 and not question_part6_7: score += 1
    if question_part6_9 and not question_part6_8: score += 1

    # Gán vào session_state để dùng bên ngoài
    st.session_state.score = score

    return score

# Nút tính toán
if st.button("Tổng điểm MMSE:"):
    score = calculate_score()
    st.session_state.button_clicked = True

# Hiển thị kết quả chỉ khi nút được nhấn
if st.session_state.button_clicked:
    st.success(f"Điểm số của bạn: {st.session_state.score}")
    # Đánh giá mức độ
    score = st.session_state.score
    if 24 <= score <= 30:
        st.success("Không suy giảm nhận thức")
    elif 19 <= score <= 23:
        st.warning("Suy giảm nhận thức nhẹ; (Có thể cần giám sát, hỗ trợ)")
    elif 10 <= score <= 18:
        st.error("Suy giảm nhận thức trung bình; (Khiếm khuyết rõ, có thể cần giám sát 24/24h)")
    else:
        st.error("Suy giảm nhận thức nghiêm trọng; (Khiếm khuyết nặng, cần giám sát 24 giờ và trợ giúp trong sinh hoạt hàng ngày)")

# Sử dụng giá trị `score` bên ngoài
final_score = st.session_state.get("score", 0)

    
# ADL
# Hàm tính điểm cho mỗi câu hỏi
def calculate_points(option):
    if option == "Independent":
        return 10
    elif option == "Needs help":
        return 5
    elif option == "Unable":
        return 0
    elif option =="Continent":
        return 10
    elif option=="Occasional accident":
        return 5
    elif option=="Incontinent (or needs to be given enemas)":
        return 0
    return 0

st.markdown('# ADL (Thang đo đánh giá khả năng tự chăm sóc bản thân):')
st.info('Với 10 câu hỏi, những câu hỏi này đều là về những vận động cơ bản của con người, chỉ số ADL sẽ cho biết tỉ lệ phụ thuộc hay không phụ thuộc của bệnh nhân')
# Câu hỏi 1: Feeding
feeding = st.radio("Khả năng tự ăn uống", ("Tự làm được", "Cần hỗ trợ", "Không có khả năng"), index=0)
feeding_score = calculate_points(feeding)

# Câu hỏi 2: Bathing
bathing = st.radio("Tự tắm rửa", ("Tự làm được", "Không có khả năng"), index=0)
if bathing == "Tự làm được":
    bathing_score = 5
elif bathing == "Không có khả năng":
    bathing_score = 0

# Câu hỏi 3: Grooming
grooming = st.radio("Tự chăm sóc bản thân", ("Tự làm được", "Không có khả năng"), index=0)
if grooming == "Tự làm được":
    grooming_score=5
if grooming=="Không có khả năng":
    grooming_score=0

# Câu hỏi 4: Dressing
dressing = st.radio("Mặc quần áo", ("Tự làm được", "Cần hỗ trợ", "Không có khả năng"), index=0)
dressing_score = calculate_points(dressing)
# Câu hỏi 5:Bowel control
bowel_control= st.radio("Kiểm soát đại tiện",("Tự kiểm soát","Thỉnh thoảng gặp sự cố","Không kiểm soát được (hoặc cần được hỗ trợ)"),index=0)
bowel_control_score=calculate_points(bowel_control)
# Câu hỏi 6: Bladder control
bladder_control=st.radio("Kiểm soát bàng quang",("Tự kiểm soát","Thỉnh thoảng gặp sự cố","Không kiểm soát được (hoặc cần được hỗ trợ)"),index=0)
bladder_control_score=calculate_points(bladder_control)
# Câu hỏi 7: Toilet use
toilet_use=st.radio("Sử dụng toilet",("Tự làm được", "Cần hỗ trợ", "Không có khả năng"),index=0)
toilet_use_score=calculate_points(toilet_use)
# Câu hỏi 8: Transfers (bed to chair and back)
transfer=st.radio("Khả năng di chuyển từ giường sang ghế",("Tự làm được", "Cần giúp đỡ nhỏ (bằng lời nói hoặc thể chất)", "Cần giúp đỡ lớn (1-2 người, thể chất), có thể ngồi","Không có khả năng"),index=0)
if transfer =="Tự làm được":
    transfer_score=15
if transfer =="Cần giúp đỡ nhỏ (bằng lời nói hoặc thể chất)":
    transfer_score=10
if transfer =="Cần giúp đỡ lớn (1-2 người, thể chất), có thể ngồi":
    transfer_score=5
if transfer =="Không có khả năng":
    transfer_score=0
# Câu hỏi 9: Mobility on level surfaces
mobility_on_level_surfaces=st.radio("Khả năng di chuyển",("Tự làm được (nhưng có thể sử dụng bất kỳ dụng cụ hỗ trợ nào, ví dụ: gậy) >50 yards","Đi bộ với sự giúp đỡ của một người (bằng lời nói hoặc thể chất) >50 yards","Xe lăn tự lập, bao gồm cả góc cua, >50 yards","Không di chuyển được hoặc <50 yards"),index=0)
if mobility_on_level_surfaces =="Tự làm được (nhưng có thể sử dụng bất kỳ dụng cụ hỗ trợ nào, ví dụ: gậy) >50 yards":
    mobility_on_level_surfaces_score=15
if mobility_on_level_surfaces =="Đi bộ với sự giúp đỡ của một người (bằng lời nói hoặc thể chất) >50 yards":
    mobility_on_level_surfaces_score=10
if mobility_on_level_surfaces =="Xe lăn tự lập, bao gồm cả góc cua, >50 yards":
    mobility_on_level_surfaces_score=5
if mobility_on_level_surfaces =="Không di chuyển được hoặc <50 yards":
    mobility_on_level_surfaces_score=0
# Câu 10:Stairs
stairs=st.radio("Di chuyển trên cầu thang",("Tự làm được","Cần hỗ trợ (bằng lời nói, thể chất, mang dụng cụ hỗ trợ)","Không có khả năng"),index=0)
if stairs=="Tự làm được":
    stairs_score=10
if stairs=="Cần hỗ trợ (bằng lời nói, thể chất, mang dụng cụ hỗ trợ)":
    stairs_score=5
if stairs=="Không có khả năng":
    stairs_score=0
total_adl_score=(feeding_score + bathing_score + grooming_score + dressing_score+bowel_control_score+bladder_control_score+toilet_use_score+transfer_score+mobility_on_level_surfaces_score+stairs_score)
st.subheader(f"Tổng điểm: {total_adl_score}")
if 0<= total_adl_score <20:
    st.write("Hoàn toàn phụ thuộc")
elif 20<= total_adl_score <40:
    st.write("Đa phần phụ thuộc")
elif 40<= total_adl_score <60:
    st.write("Bán phụ thuộc")
elif 60<= total_adl_score <80:
    st.write("Ít phụ thuộc")
elif 80<= total_adl_score <=100:
    st.write("Hoàn toàn tự lập")
# Confusion
## Memory Complaint
st.markdown("# Bộ câu hỏi kiểm tra trí nhớ của bệnh nhân (Memory Complaints)")
memory_complaint_1 = st.radio("1. Bạn có thường xuyên quên đi đây là ngày nào trong tuần không ?", ("Có", "Không"))
memory_complaint_2 = st.radio("2. Đôi khi bạn tìm kiếm một vật gì đó, bạn có quên mất nó là vật gì không?", ("Có", "Không"))
memory_complaint_3 = st.radio("3. Gia đình và bạn bè bạn có nghĩ trí nhớ bạn đã kém đi so với trước không?", ("Có", "Không"))
memory_complaint_4 = st.radio("4. Bạn có hay quên tên bạn bè/ người thân xung quanh không?", ("Có", "Không"))
memory_complaint_5 = st.radio("5. Bạn có gặp khó khăn khi cộng hai số lại với nhau mà không cần  viết ra giấy không ?", ("Có", "Không"))
memory_complaint_6 = st.radio("6. Bạn có thường xuyên bị lỡ hẹn do quên mất chúng không?", ("Có", "Không"))
memory_complaint_7 = st.radio("7. Bạn có hiếm khi thấy tràn trề năng lượng không?", ("Có", "Không"))
memory_complaint_8 = st.radio("8. Bạn có cảm thấy khó tập trung trong một thười gian ngắn không?", ("Có", "Không"))
memory_complaint_9 = st.radio("9. Bạn có thường đặt nhầm chìa khóa, và khi tìm lại, bạn có quên mất là mình đã đặt nó ở đó không?", ("Có", "Không"))
memory_complaint_10 = st.radio("10. Bạn có hay khó chịu với những vấn đề rất nhỏ nhặt không?", ("Có", "Không"))
memory_complaint_11 = st.radio("11. Bạn có hay vô tình lặp lại những gì đã nói không?", ("Có", "Không"))
memory_complaint_12 = st.radio("12. Bạn có đôi khi bị lạc đường dù cho bạn đã đến nơi này trước đó không?", ("Có", "Không"))
memory_complaint_13 = st.radio("13. Bạn có đôi khi quên đi những gì mình chuẩn bị nói không?", ("Có", "Không"))
memory_complaint_14 = st.radio("14. Bạn có hay dùng caffein khi muốn đầu óc minh mẫn không?", ("Có", "Không"))
memory_complaint_15 = st.radio("15. Bạn có cảm thấy học một việc gì đó khó hơn so với trước khia không?", ("Có", "Không"))
# Đánh giá các câu trả lời và hiển thị kết quả
total_complaints = sum([memory_complaint_1 == "Có", memory_complaint_2 == "Có", memory_complaint_3 == "Có",
                        memory_complaint_4 == "Có", memory_complaint_5 == "Có", memory_complaint_6 == "Có",
                        memory_complaint_7 == "Có", memory_complaint_8 == "Có", memory_complaint_9 == "Có",
                        memory_complaint_10 == "Có",memory_complaint_11 == "Có", memory_complaint_12 == "Có",memory_complaint_13 == "Có"
                       ,memory_complaint_14 == "Có",memory_complaint_15 == "Có"])

st.subheader(f"Tổng số vấn đề trí nhớ: {total_complaints} trên 15")
if 5<=total_complaints<=8:
    st.success("Não bộ của bạn hoạt động bình thường")
if 9<=total_complaints<=12:
    st.info(" Não của bạn đang trong tình trạng nguy hiểm. Hãy kiểm tra chế độ ăn uống của bạn ngay hôm nay.Bạn có thể giảm thiểu sự suy giảm trí não và mất trí nhớ bằng cách sử dụng vitamin, thực phẩm tốt cho não, thảo dược, các kỹ thuật yoga và thiền, cùng với việc sử dụng thuốc phù hợp.")
if total_complaints>= 15:
    st.warning("Não của bạn đang cạn kiệt năng lượng. Bạn nên đi khám bác sĩ. Bạn có thể hồi phục não bộ bằng cách sử dụng thực phẩm, vitamin, thảo dược, bài tập và thuốc.")
## Behavior Problems
st.markdown("# Behavioral Problems (Vấn đề hành vi):")
st.info("Bác sĩ đánh giá các hành vi của bệnh nhân dựa trên quan sát.")

behavior_1 = st.radio("1. Bệnh nhân có biểu hiện kích động hoặc cáu gắt thường xuyên không?", ("Có", "Không"))
behavior_2 = st.radio("2. Bệnh nhân có xu hướng đi lại liên tục mà không có mục đích rõ ràng không?", ("Có", "Không"))
behavior_3 = st.radio("3. Bệnh nhân có hành vi lặp đi lặp lại (ví dụ: lặp lại câu nói, hành động) không?", ("Có", "Không"))
behavior_4 = st.radio("4. Bệnh nhân có biểu hiện bối rối hoặc không nhận thức được vị trí hiện tại không?", ("Có", "Không"))
behavior_5 = st.radio("5. Bệnh nhân có gặp khó khăn trong việc kiểm soát cảm xúc không?", ("Có", "Không"))
behavior_6 = st.radio("6. Bệnh nhân có biểu hiện né tránh xã hội hoặc ít giao tiếp hơn trước đây không?", ("Có", "Không"))
behavior_7 = st.radio("7. Bệnh nhân có hay nghi ngờ người khác hoặc có biểu hiện ảo giác không?", ("Có", "Không"))
behavior_8 = st.radio("8. Bệnh nhân có mất khả năng thực hiện các hoạt động hàng ngày một cách đột ngột không?", ("Có", "Không"))
behavior_9 = st.radio("9. Bệnh nhân có hay quên vị trí để đồ hoặc cất đồ ở những nơi không hợp lý không?", ("Có", "Không"))
behavior_10 = st.radio("10. Bệnh nhân có biểu hiện không hiểu các chỉ dẫn đơn giản không?", ("Có", "Không"))

# Đánh giá các câu trả lời và hiển thị kết quả
total_behavior_issues = sum([behavior_1 == "Có", behavior_2 == "Có", behavior_3 == "Có", behavior_4 == "Có",
                             behavior_5 == "Có", behavior_6 == "Có", behavior_7 == "Có", behavior_8 == "Có",
                             behavior_9 == "Có", behavior_10 == "Có"])

st.subheader(f"Tổng số vấn đề hành vi: {total_behavior_issues} trên 10")

# Phân loại mức độ nghiêm trọng
if 0 <= total_behavior_issues <= 6:
    st.write("Hành vi của bệnh nhân ở mức bình thường.")
elif 7 <= total_behavior_issues <= 10:
    st.write("Hành vi của bệnh nhân có biểu hiện bất thường nghiêm trọng.")

# Biến đổi nhị phân
if total_behavior_issues >=7:
    total_behavior_issues_binary=1
else:
    total_behavior_issues_binary=0
if total_complaints>=10:
    total_complaints_binary=1
else:
    total_complaints_binary=0
expected_features=["MMSE",'FunctionalAssessment','MemoryComplaints','BehavioralProblems','ADL']

user_input=[final_score,fas,total_complaints_binary,total_behavior_issues_binary,total_adl_score/10]
input_data = pd.DataFrame([user_input], columns=expected_features)

# Khi nhấn nút, thực hiện dự đoán
if st.button("Dự đoán khả năng mắc bệnh Alzheimer"):
    # Dự đoán kết quả
    prediction = model.predict(input_data)

    # Dự đoán xác suất
# Dự đoán xác suất
    probabilities = model.predict_proba(input_data)

# Hiển thị kết quả dự đoán
    if 0.4 <= probabilities[0][1] <= 0.6:
        st.write("Kết quả dự đoán:", "Có khả năng mắc bệnh Alzheimer")
    elif probabilities[0][1] < 0.4:
        st.write("Kết quả dự đoán:", "Không mắc bệnh Alzheimer")
    else:
        st.write("Kết quả dự đoán:", "Có khả năng đã mắc bệnh Alzheimer")

# Hiển thị xác suất
    st.write(f"Xác suất: {probabilities[0][1]:.2f}")
