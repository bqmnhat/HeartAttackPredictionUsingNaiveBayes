# Heart Attack and Diabetes Prediction Model using Naive Bayes

- Author:
    - Nhat Bui Quoc Minh (Leader, A.I developer, web application developer, markdown writer)
    - An Nguyen Pham Quoc (A.I developer, markdown writer)
    - Choi Won Seok (web application developer)
    - Thinh Duong Ngoc (web application developer)
    - Phi Nguyen Nhat (markdown writer)
- Tran Dai Nghia highschool for the gifted - class of 12CTin (2021 - 2024)

Content
=================
* 1 [Giới thiệu](#1-Giới-thiệu)
    * 1.1 [Hiện trạng bệnh tiểu đường](#11-Hiện-trạng-bệnh-tiểu-đường)
    * 1.2 [Hiện trạng nhồi máu cơ tim](#12-Hiện-trạng-nhồi-máu-cơ-tim)
    * 1.3 [Mục tiêu của sản phẩm](#13-Mục-tiêu-của-sản-phẩm)
* 2 [Dữ liệu](#2-Dữ-liệu)
    * 2.1 [Các trường dữ liệu cần thu thập](#21-Các-trường-dữ-liệu)
    * 2.2 [Dữ liệu thô](#22-Dữ-liệu-thô)
* 3 [Xây dựng mô hình sản phẩm](#3-Xây-dựng-mô-hình-sản-phẩm)
    * 3.1 [Lưu đồ quá trình xây dựng](#31-Lưu-đồ-quá-trình-xây-dựng)
    * 3.2 [Xây dựng mô hình](##32-Xây-dựng-mô-hình)
    * 3.3 [Huấn luyện model](#33-Huấn-luyện-model)
    * 3.4 [Trích xuất model](#34-Trích-xuất-model)
    * 3.5 [Xây dựng ứng dụng web](#35-Xây-dựng-ứng-dụng-Web)
    * 3.6 [Thử nghiệm về quá khớp và chưa khớp](#36-Thử-nghiệm-về-quá-khớp-và-chưa-khớp)
* 4 [Kết luận](#4-Kết-luận)
    * 4.1 [Kết quả đạt được](#41-Kết-quả-đạt-được)
    * 4.2 [Đánh giá](#42-Đánh-giá)

## 1. Giới thiệu

Theo kết quả điều tra năm 2012, tỉ lệ đái tháo đường ở nước ta là 5,7%. Nhưng vào năm 2022, tỉ lệ trên là 7,3%. Có thể thấy, trong vòng mười năm tỉ lệ đái tháo đường đã tăng nhanh chóng mặt với 1,6% và trở thành một hiểm họa lớn ảnh hưởng đến sức khỏe của nhiều người.
<div style="text-align: center;" markdown="1">

***Biểu đồ thể hiện tỉ lệ mắc bệnh đái tháo đường trong giai đoạn 2012-2022 ở nước ta***
![image](https://hackmd.io/_uploads/HyTlcPYSa.png)

</div>

Đi kèm với đái tháo đường là nguy cơ cao dẫn tới đột quỵ. Khi mà mỗi năm có khoảng 200.000 ca, đạt tỉ lệ tử vong là 20%.

Và nhằm mục đích nâng cao chất lượng sống và sức khỏe của người Việt Nam cũng như là thế giới. Mô hình chuẩn đoán nhồi máu cơ tim và tiểu đường được phát triển nhằm cải thiện khả năng dự đoán sớm và can thiệp kịp thời vào hai bệnh lý nguy hiểm này. Mô hình chẩn đoán có thể đóng vai trò quan trọng trong việc xác định nguy cơ, chẩn đoán và điều trị.

### 1.1 Hiện trạng bệnh tiểu đường 
#### 1.1.1 Việt Nam

Trước những năm 2000, tỷ lệ mắc bệnh tiểu đường của nam và nữ đều dưới 5%. Trong đó phụ nữ cao hơn nam giới khoảng 1% và bắt đầu tăng liên tục đạt khoảng 5% vào năm 2000 thì dừng lại. Còn về phần nam giới tăng liên tục trong suốt giai đoạn và cao hơn nữ giới vào năm 2010. 
<div style="text-align: center;" markdown="1">

***Biểu đồ thể hiện tốc độ tăng tỷ lệ mắc bệnh tiểu đường trong giai đoạn 1960-2014 ở nước ta***

![image](https://hackmd.io/_uploads/r14HyuKSp.png)

</div>

Có thể thấy tỉ lệ mắc bệnh tiểu đường ở cả hai giới đều có xu hướng tăng. Trước năm 2000 thì tỷ lệ nữ mắc bệnh cao hơn còn sau năm 2000 thì nam giới chiếm ưu thế hơn.

Đây là một tỷ lệ tăng rất nhanh và đáng báo động, cần phải có các biện pháp giải quyết ngay.

Mặc dù vậy, tình hình chẩn đoán và điều trị bệnh tại Việt Nam vẫn còn ảm đạm. Cụ thể do người bệnh không phát hiện được các dấu hiệu khi bệnh nhẹ hoặc xem nhẹ các dấu hiệu bệnh. Ngoài ra, việc di chuyển đến bệnh viện để được khám chữa bệnh và chẩn đoán định kỳ thường **tốn nhiều thời gian và công sức**. Điều này dẫn đến **phần lớn bệnh nhân không được chẩn đoán hay thậm chí là không được điều trị.**

<div style="text-align: center;" markdown="1">

![image](https://hackmd.io/_uploads/S1X3QaYB6.png)

</div>

Dựa vào biểu đồ trên, có thể thấy **số lượng bệnh nhân vẫn chưa được chẩn đoán chiếm tỉ lệ lớn áp đảo với 68,9%**. Qua đó, ta thấy rằng hiện nay vẫn còn nhiều cá nhân chưa được chẩn đoán và điều trị kịp thời tại Việt Nam. 

#### 1.1.2 Thế giới
Không chỉ ở Việt Nam, thực trang căn bệnh tiểu đường ở nhiều nước cũng không khả quan:
<div style="text-align: center;" markdown="1">

![image](https://hackmd.io/_uploads/HJHnqaFS6.png)

</div>

Tiểu đường từng được xem là căn bệnh của Châu Âu nhưng giờ đây đang có xu hướng lan sang Châu Á. Khi mà các khu vực ở Châu Á đặc biệt như Tây Thái Bình Dương, Đông Nam Á và Trung Đông đều chiếm tỉ trọng lớn và chiếm 68,7%. Tây Thái Bình Dường chiểm tỉ trọng lớn nhất với 206 triệu người. Việc số lượng người mắc bệnh chủ yếu từ Châu Á có thể do dân số đông, ô nhiễm thực phẩm,...
<div style="text-align: center;" markdown="1">

![image](https://hackmd.io/_uploads/r1laZ0tB6.png)

</div>

Có thể thấy số lượng người mắc tiểu đường đang tăng liên tục trên toàn thế giới. Dự đoán tiếp tục tăng lên 578 triệu người vào năm 2030 và 700 triệu người vào năm 2045. 
### 1.2 Hiện trạng nhồi máu cơ tim
#### 1.2.1 Việt Nam 

Nhồi máu cơ tim cũng như là các bệnh về tim mạch đang là một trong những nguyên nhân hàng đầu tử vong do bệnh khi chiếm đến 33%.
<div style="text-align: center;" markdown="1">

![image](https://hackmd.io/_uploads/Sk29XCKST.png)

</div>

Nhồi máu cơ tim là một bệnh lý nguy hiểm, nếu không được cấp cứu kịp thời nhằm khôi phục lưu lượng máu nhanh chóng, có thể gây tổn thương tim vĩnh viễn và tử vong.

Trước đây, bệnh thường gặp ở người cao tuổi, tuy nhiên nhiều số liệu cho thấy nhồi máu cơ tim đang trẻ hóa. Theo một số liệu thống kê tại các bệnh viện lớn tỷ lệ nhồi máu cơ tim ở người trẻ đã tăng lên đến 10,5% và rất trẻ là 1,8%.

#### 1.2.2 Thế giới 

Trên thế giới mỗi năm có 2,5 triệu người chết do bệnh nhồi máu cơ tim, trong đó 25% chết trong giai đoạn cấp tính của bệnh. Trong vòng năm sau đó chết thêm 5% – 10% nữa.Mỗi năm có khoảng 635.000 trường hợp nhồi máu mới xuất hiện và 280.000 trường hợp nhồi máu tái phát, 150.000 nhồi máu im lặng. Ước tính cứ mỗi 34 giây là có một trường hợp nhồi máu xuất hiện, 1 phút là có một trường hợp tử vong.

### 1.3 Mục tiêu của sản phẩm:

Từ những tình trạng thực tiễn được thể hiện bằng số liệu, ta rút ra kết luận:

Cần một mô hình chẩn đoán và khám chữa bệnh hiệu quả hơn. Việc phát triển mô hình chẩn đoán bệnh tiểu đường và nhồi máu cơ tim ở Việt Nam là một giải pháp mở ra con đường phát triển các hệ thống khám bệnh từ xa cho 2 căn bệnh trên nói riêng và các bệnh lý nói chung. Các bác sĩ vừa được cung cấp platform khám bệnh từ xa, vừa có thể dựa vào sự hỗ trợ của trí tuệ nhân tạo (AI) để chẩn đoán. Điều này tiết kiệm thời gian và công sức cho cả bệnh nhân và bác sĩ, làm giảm các khoảng thời gian di chuyển đến bệnh viện và có thể khám bệnh mọi lúc mọi nơi.

**Mục tiêu xây dựng** sản phẩm sẽ gồm các chứng năng sau đây:
- Quản lý tài khoản và thông tin của từng người dùng: yêu cầu database
- **Hỗ trợ chẩn đoán bệnh nhồi máu cơ tim và bệnh tiểu đường: model AI**
- Cho phép bác sĩ kết nối với bệnh nhân để khám chữa bệnh
- Sử dụng đồng hồ thông minh để đo các thông số sức khỏe cần thiết

Bài viết dưới đây sẽ tập trung vào phần trí tuệ nhân tạo (AI) hỗ trợ các bác sĩ chẩn đoán bệnh.

## 2. Dữ liệu

### 2.1 Các trường dữ liệu

Nhồi máu cơ tim (Heart Attack) là tình trạng xảy ra khi 1 hoặc cả 2 nhánh động mạch vành bị tắc nghẽn đột ngột, dẫn đến tình trạng cơ tim không nhận đủ máu, và có khả năng bị hoại tử cơ tim. Có thể có một hoặc nhiều nguyên nhân cùng tác động làm ảnh hưởng đến lưu thông trong vòng tuần hoàn máu của cơ thể. Ví dụ: Cholesterol trong máu làm tắc nghẽn mạch máu.

Ngoài ra, các dấu hiệu suy giảm sức khỏe thường xuất hiện trước khi nhồi máu cơ tim đột ngột diễn ra. Tuy nhiên, do các dấu hiệu đó không thể hiện ra rõ rệt và không có mối liên hệ gần gũi với hệ tim nhưng ảnh hưởng đến hệ tim thông qua ảnh hưởng đến các hệ thống khác trong cơ thể nên người bệnh thường bỏ qua. Do đó, để chẩn đoán khả năng một bệnh nhân bị nhồi máu cơ tim, ta cần thu thập rất nhiều thông số sức khỏe của bệnh nhân. Vì vậy, đối với model AI của chúng tôi, các trường dữ liệu sẽ tương ứng với các số liệu sức khỏe. Trong đó, gồm:

- Trường dữ liệu **sinh học**: tuổi, giới tính, chiều cao, cân nặng, huyết áp, tiền sử bị cao huyết áp, tiền sử bị bệnh tim và nồng độ cholesterol trong máu
- Trường dữ liệu **di truyền**:  bệnh tim di truyền
- Trường dữ liệu **vật lý**: mức độ tập thể dục
- Trường dữ liệu **khác**: tình trạng sử dụng thuốc lá, tình trạng sử dụng đồ uống có cồn, mức độ ăn "xanh" và mức độ stress

Thông qua các trường dữ liệu dùng để chẩn đoán nhồi máu cơ tim, ta còn có thể mở rộng thêm sang vấn đề chẩn đoán khả năng bệnh nhân bị bệnh tiểu đường. Điều này là bởi 2 yếu tố sau:
- "**Khả năng bị nhồi máu cơ tim"** là một trong những yếu tố có ảnh hưởng lớn đối với việc chẩn đoán bệnh tiểu đường nhưng lại khó thu thập dữ liệu từ bệnh nhân. 
- Rất nhiều biểu hiện của bệnh nhồi máu cơ tim trùng với biểu hiện của bệnh tiểu đường, được thể hiện qua các thông số sức khỏe tương tự nhau.

Vì vậy, ta có thể sử dụng bộ dữ liệu thu thập được và kết quả dự đoán khả năng bị nhồi máu cơ tim để dự đoán khả năng bị bệnh tiểu đường của bệnh nhân.

### 2.2 Dữ liệu thô

Dưới đây là bảng biểu thị 15 trường dữ liệu của bộ dữ liệu thô, được thu thập và tổng hợp từ bộ dữ liệu về nhồi máu cơ tim trên trang web Kaggle.com và bộ dữ liệu về sức khỏe của chính phủ Mỹ trên data.cdc.gov:

![DiabetesBeforePreprocess_NoMarks2](https://hackmd.io/_uploads/HkUacHoHa.png)

Trong đó, 15 trường gồm:
- Age: Tuổi
- Gender: Giới tính (0 là nữ, 1 là nam)
- Hypertension: Cao huyết áp
- Heart Disease: Tình trạng bệnh tim
- Average Glucose Level: Nồng độ đường trong máu
- BMI: Chỉ số BMI
- Smoking Status: Tình trạng hút thuốc lá
- Alcohol intake: Tình trạng tiêu thụ đồ uống có cồn
- Physical Activity: Mức độ thường xuyên chơi thể thao 
- Previous Heart Disease: Tiền sử bệnh tim
- Family stroke history: Tiền sử dòng họ bị đột quỵ
- Dietary Habits: Mức độ ăn kiên/ăn "xanh"
- Stress Levels: Mức độ stress
- Blood Pressure Levels: Huyết áp
- Cholesterol Levels: Nồng độ cholesterol

## 3 Xây dựng mô hình sản phẩm
### 3.1 Lưu đồ quá trình xây dựng:
![HeartAttackAndDiabetes.drawio](https://hackmd.io/_uploads/HJpjsI5S6.png)

### 3.2 Xây dựng mô hình

Trong trường hợp bài toán của chúng ta, ta sẽ xử lý dữ liệu dựa vào mối quan hệ giữa các trường dữ liệu nhằm loại bỏ các trường không ổn định hoặc tách các trường là cha của nhiều thành phần trường dữ liệu nhỏ hơn để dễ dàng huấn luyện model.

Ngoài ra, ta sẽ huấn luyện nhiều model khác nhau và xét tính chính xác của tất cả các model để từ đó chọn ra model mạnh nhất hoặc hướng xây dựng model phù hợp cho sản phẩm.

#### 3.2.1 Tiền xử lý dữ liệu
Trong các trường dữ liệu ban đầu, ta thấy rằng 2 trường dữ liệu "Blood pressure level" và "Cholesterol Level" đang chứa 2 thông số (2 trường dữ liệu con) bên trong. Điều này sẽ gây khó khăn trong việc trích xuất các feature cần thiết cho việc huấn luyện model. Do đó, 2 trường dữ liệu này cần được phân tách ra thành các trường dữ liệu có viền màu xanh lá cây như bên dưới:

![DiabetesBeforePreprocess_HaveAverageGlucose](https://hackmd.io/_uploads/By0r7GcBT.png)

Ngoài ra ảnh hưởng của thông số sức khỏe được ghi nhận trong trường dữ liệu "Average Glucose Level" (Lượng đường trung bình trong máu) sẽ tương đương với việc kết hợp 2 trường dữ liệu "Gender", "HDL Cholesterol Level" và "LDL Cholesterol Level". Điều này cũng được chứng minh trong publication: 
<div style="text-align: center;" markdown="1">

![AGLCorrelationWithGenderAndCholLevel_Proof](https://hackmd.io/_uploads/BJrJfzqra.png)
*(Nguồn:https://www.researchgate.net/publication/341653979_Correlating_the_Cholesterol_Levels_to_Glucose_for_Men_and_Women)*

</div>

Dựa vào điều này, ta hoàn toàn có thể thay thế trường dữ liệu "Average Glucose Level" bằng việc sử dụng trường dữ liệu "Gender" và 2 trường dữ liệu về "Cholesterol" để chẩn đoán:

![DiabetesBeforePreprocess_AverageGlucoseCorrelationWithGenderAndCholLevels](https://hackmd.io/_uploads/SJuhXfqHT.png)

Sau khi hoàn thành tinh chỉnh các trường dữ liệu, nhận thấy rằng dữ liệu chứa toàn dữ liệu dạng số, ta cần xác định đặc trưng phân phối của dữ liệu để xây dựng model AI phù hợp.

Tuy nhiên, việc phân tích dữ liệu để chỉ ra đặc trưng phân phối bằng các bảng số liệu chằn chịt các con số như trên là rất khó. Vì vậy, ta cần mô hình hóa và đồ thị hóa các trường dữ liệu và xét từng trường dữ liệu rồi sau đó gộp đặc trưng phân phối lại để lựa chọn hướng xây dựng model Naive Bayes thích hợp (Gausian hay Multinomial):

![CodePlotAll](https://hackmd.io/_uploads/HysKLf5Ba.png)
![AllGraphs1](https://hackmd.io/_uploads/rkvX_G5Hp.png)
![AllGraphs2](https://hackmd.io/_uploads/BkCXdzqST.png)
![AllGraphs3](https://hackmd.io/_uploads/BJmV_fqBa.png)
![AllGraphs4](https://hackmd.io/_uploads/B1UNufqrT.png)

Thông qua các biểu đồ trên, ta nhận thấy rằng sự phân phối dữ liệu của các trường gồm có 2 kiểu:
- Phân phối rời rạc (Multinomial):
    - Age
    - Average Glucose Level
    - BMI
    - Smoking Status
    - Alcohol Intake
    - Physical Activity
    - Dietary Habits
    - Stress Levels
    - Các trường Blood Pressure
    - Các trường Cholesterol
- Phân phối nhị phân (Binomial):
    - Gender
    - Hypertension
    - Heart Disease
    - Previous Heart Disease
    - Family Stroke History

#### 3.2.2 Lựa chọn model AI phù hợp: Naive Bayes

Từ kết quả trên, ta rút ra kết luận rằng phát triển model chẩn đoán theo hướng **Naive Bayes MultinomialNB** sẽ là một hướng xây dựng model phù cho bộ dữ liệu mà chúng ta thu thập được. 

Thật vậy, khi huấn luyện và kiểm thử bộ dữ liệu đã qua tiền xử lý với những mô hình khác nhau (Naive Bayes Multinomial, Random Forest, Logistic Regression và SVM), ta thấy rằng Naive Bayes Multinomial có **accuracy score cao hơn mức trung bình (Mean)**:
![Evaluation](https://hackmd.io/_uploads/Byp6-P9Sp.png)

#### 3.2.3 Cơ sở áp dụng Naive Bayes
Phân loại Bayes là một phần trong lĩnh vực học máy và thống kê, cho phép dự đoán xác suất của một điểm dữ liệu thuộc vào một lớp cụ thể. Nó dựa trên định lý Bayes để ước lượng xác suất của sự kiện dựa trên thông tin trước đó:

![naive](https://hackmd.io/_uploads/rJbg10tSp.png)

Dù có nhiều dạng và loại, các thuật toán phân lớp Bayes đều đang được ứng dụng rộng rãi trong lĩnh vực khoa học máy tính, đặc biệt trong trí tuệ nhân tạo. 

#### Các thông số cần thiết

Trong thuật toán Naive Bayes, có một số thông số quan trọng cần thiết để xây dựng mô hình và thực hiện phân loại:

 - **Xác suất tiên nghiệm** của các lớp (Prior Probability): ước lượng dựa trên tần suất xuất hiện của các lớp trong tập dữ liệu huấn luyện.
 - **Các đặc trưng** của bộ dữ liệu bài toán
- **Xác suất có điều kiện của các đặc trưng cho từng lớp** (Conditional Probability): Xác suất có điều kiện của từng đặc trưng tương ứng với từng lớp trong bài toán phân loại. 
 - Tham số làm mịn (Laplace Smoothing): **Đặc biệt trong Multinomial Naive Bayes**, thường sử dụng tham số làm mịn Laplace để tránh việc xác suất có điều kiện bằng 0 trong trường hợp một đặc trưng không xuất hiện trong tập huấn luyện.

#### Áp dụng Naive Bayes vào bài toán

Áp dụng thuật toán Naive Bayes vào bài toán chẩn đoán bệnh nhồi máu cơ tim và chẩn đoán bệnh tiểu đường, ta nhận ra rằng:

- **Xác suất tiên nghiệm** trong bài toán là tần suất xuất hiện của số bệnh nhân bị bệnh và số bệnh nhân không bị bệnh trong tập dữ liệu. Ví dụ trong bộ dữ liệu nhồi máu cơ tim:

<div style="text-align: center;" markdown="1">

![PercentageOfClassesInDataset](https://hackmd.io/_uploads/SkWwbIsBp.png)

</div>

- **Các đặc trưng** là các thông số sức khỏe
- **Xác suất có điều kiện của các đặc trưng cho từng lớp** là xác suất có điều kiện của từng loại thông số sức khỏe tương ứng với tình trạng bị bệnh và không bị bệnh. Ví dụ:

<div style="text-align: center;" markdown="1">

![AlcoholIntakeWhenPatientHaveHeartAttack](https://hackmd.io/_uploads/B1uFPIjBp.png)

</div>

### 3.3 Huấn luyện model

Đầu tiên, ta cần chia tập dữ liệu lớn thành 2 tập dữ liệu con là train (dùng để huấn luyện) và test (dùng để kiểm thử) sao cho tỉ lệ train:test là gần tỉ lệ 7:3 nhất (chứng minh trong paper "Optimal Ratio for Data Splitting" của V. Roshan Joseph):
<div style="text-align: center;" markdown="1">

![Split1](https://hackmd.io/_uploads/HyNqQPcB6.png)
![Split2](https://hackmd.io/_uploads/r1vqQP5rp.png)

</div>

### 3.4 Trích xuất model

Sau khi huấn luyện thành công model, việc tiếp theo ta cần làm là trích xuất model để sử dụng trong các chương trình của ứng dụng.

Trước khi được trích xuất, các model chỉ tồn tại trong các chương trình python. Và khi chương trình python bị đóng lại thì sẽ kéo theo việc các thông số của model bị mất đi và đồng thời ta cũng **không thể sử dụng các model này trong các chương trình khác**. **Vì vậy, cần một phương pháp để trích xuất và lưu trữ model từ code python xuống đĩa cứng máy tính.**

Phương pháp này được gọi là **serialization**. Khi ta serialize một python object, chúng ta chuyển object này từ một kiểu dữ liệu phức tạp thành một **dòng chảy các byte (byte stream)**. Byte stream này được thiết lập với tư tưởng tương tự như việc ta **encode** dữ liệu. Tuy nhiên, điều đặc biệt là bye stream này phải được thiết lập theo một trình tự đặc biệt để có thể **decode** trở lại thành model AI khi ta cần sử dụng. 

<div style="text-align: center;" markdown="1">

***Sơ đồ serialization một python object***
![Serialization](https://hackmd.io/_uploads/HJTXuvqBp.png)

</div>

Ngôn ngữ python hỗ trợ rất nhiều các thư viện có khả năng serialize python object. Trong số đó, thử viện **pickle** thường được sử dụng nhiều nhất vì các hàm đơn giản, tiện lợi và nhanh chóng.

Có 2 hàm mà ta cần lưu ý:

1. Thủ tục **pickle.dump(model, file)**: serialize, tạo byte stream của "model" rồi lưu vào "file"

<div style="text-align: center;" markdown="1">

![dumping](https://hackmd.io/_uploads/H11oIvcSp.png)

</div>

2. Hàm **load(file)**: giá trị trả về là một python object được deserialize từ byte stream trong "file"

<div style="text-align: center;" markdown="1">

![Load](https://hackmd.io/_uploads/H12pIvcSa.png)

</div>

Sau khi load, ta có thể sử dụng model bằng lệnh **model.predict(data)** bình thường như trong chương trình python.

### 3.5 Xây dựng ứng dụng Web

Sản phẩm web app được tích hợp model AI vừa có thể chẩn đoán bệnh nhồi máu cơ tim và tiểu đường vừa được xây dựng theo hướng có khả năng tích hợp các chức năng khác nhằm mở rộng thành nền tảng khám bệnh từ xa trong tương lai.

Dưới đây là mô hình của web app:
<div style="text-align: center;" markdown="1">

![WebAppDiagram.drawio](https://hackmd.io/_uploads/HyDDAuiB6.png)

</div>

Web app gồm 3 thành phần chính:
- **Dao diện người dùng gồm**: dao diện bệnh nhân và dao diện bác sĩ (hướng phát triển tương lai)
- **Server chứa database**
- **Các thiết bị đo thông số sức khỏe** (hướng phát triển tương lai)

### 3.6 Thử nghiệm về quá khớp và chưa khớp

#### 3.6.1 Nhắc lại
Mô hình được coi là chưa khớp **(underfitting)** nếu nó chưa phù hợp với tập dữ liệu huấn luyện **(Training set)** và cả các dữ liệu mới **(Test data)** khi dự đoán. Nguyên nhân có thể là do mô hình chưa đủ độ phức tạp cần thiết để bao quát được tập dữ liệu.

Mô hình được coi là quá khớp **(overfitting)** nếu nó rất hợp lý, rất khớp với tập huấn luyện **(Training set)** nhưng khi đem ra dự đoán với dữ liệu mới **(Testing data)** thì lại không phù hợp. Nguyên nhân có thể do ta chưa đủ dữ liệu để đánh giá hoặc do mô hình của ta quá phức tạp. 
#### 3.6.2 Thử nghiệm
Nhằm thử nghiệm hai tính chất trên, ta có thể biến đối các tham số của mô hình hoặc thao túng các giá trị của dữ liệu để huấn luyện **(Data Manipulation)**. Trong phạm vi bài này, chúng tôi thực hiện hai thí nghiệm bao gồm:
- **Đặc thù hoá dữ liệu:**
    -  Nhóm tiến hành chuyển đổi tất cả các giá trị trong trường **[Diagnosis]** của tập huyến luyện thành **"1"**.

    <div style="text-align: center;" markdown="1">

    ![image](https://hackmd.io/_uploads/SkCeat2ST.png)

    </div>
    
    - Điều này dẫn đến mô hình có xu hướng dự đoán tất cả trường hợp nhập vào đều có **"Diagnosis"** là **"1"** do tập dữ liệu bị đặc thù hoá. Tuy nhiên khi đưa tập dữ liệu mới (Testing Data) vào mô hình, độ chính xác bị giảm xuống rất thấp (Do tập dữ liệu mới có bao gồm giá trị "0" và "1") 
    <div style="text-align: center;" markdown="1">

    ![image](https://hackmd.io/_uploads/rkjh6YhST.png)

    </div>

    **=> Mô hình bị quá khớp.**
    
- **Giảm bớt số trường dữ liệu:**
    -  Nhóm tiến hành lược bớt các trường tham số (15 trường được nhắc ở trên) nhằm dự đoán trường **[Diag0sis]**. Từ 15 trường, chúng tôi giảm số trường xuống thành 1 (Trường **[Age]**).

    <div style="text-align: center;" markdown="1">

    ![image](https://hackmd.io/_uploads/BygdRYnrp.png)

    </div>
    
    - Điều này dẫn đến khiến mô hình bị quá đơn giản do có ít tham số. Vì vậy, độ chính xác khi huấn luyện mô hình với tập huấn luyện và tập dữ liệu mới đều rất thấp (<60%).
    <div style="text-align: center;" markdown="1">

    ![image](https://hackmd.io/_uploads/SkyGJ52rT.png)

    </div>

    **=> Mô hình bị chưa khớp.**
    



## 4. Kết luận

### 4.1 Kết quả đạt được

Hiện tại, sản phẩm đã hoàn thành 2 chức năng:
- Quản lý tài khoản và thông tin của từng người dùng: yêu cầu database
- **Hỗ trợ chẩn đoán bệnh nhồi máu cơ tim và bệnh tiểu đường: model AI**

<div style="text-align: center;" markdown="1">

**Giao diện trang chủ**
![Homepage](https://hackmd.io/_uploads/HybYVYjHa.png)

**Giao diện đăng nhập**
![Login](https://hackmd.io/_uploads/H10pNKsSp.png)

**Giao diện chẩn đoán**
![Predict](https://hackmd.io/_uploads/BJ49O5nS6.png)
![Predict2](https://hackmd.io/_uploads/H1uq_92Ba.png)
**Biểu đồ biểu diễn thông số sức khỏe (định hướng phát triển biểu diễn thông số sức khỏe theo thời gian)**
![Predict3](https://hackmd.io/_uploads/SJncd53B6.png)


</div>

### 4.2 Đánh giá:

Mặc dù dao diện của bác sĩ và việc kết nối bác sĩ với bệnh nhân hiện vẫn đang trong quá trình phát triển như vẫn có tính thực thi cao bởi nền tảng cấu trúc dữ liệu đã có đầy đủ và hoàn toàn có thể xây dựng công cụ hỗ trợ cuộc gọi video và tích hợp thêm các thiết bị đo thông số sức khỏe thông minh để đo chỉ số sức khỏe trực tiếp tại nhà. 

Sản phẩm khi hoàn chỉnh sẽ là nền tảng tiền đề cho hệ thống khám chữa bệnh thông minh từ xa có tích hợp công nghệ trí tuệ nhân tạo và thiết bị đồng hồ thông minh để đo chỉ số sức khỏe. Điều này sẽ tạo ra một sự thay đổi tích cực trong thói quen khám bệnh của người dân, thúc đẩy người dân tham gia khám chữa bệnh nhiều hơn nhờ sự hiệu quả của sản phẩm.
