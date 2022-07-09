
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
using std::cout;
using std::endl;
using std::vector;
using namespace cv;
int main()
{
	// 从⽂件中读⼊图像
	Mat src_img1 = cv::imread("D:\\侯卓鲁\\数据\\大时相差异数据集sy_gg\\98c.1.1_gg.png", -1);
	Mat img1 = src_img1;
	//cv::resize(src_img1, img1, cv::Size(512, 512));
	Mat src_img2 = cv::imread("D:\\侯卓鲁\\数据\\大时相差异数据集sy_gg\\98c.1.2_sy.png", -1);
	Mat img2 = src_img2;
	//cv::resize(src_img2, img2, cv::Size(512, 512));
	//cv::imshow("image before", img1);
	//cv::imshow("image2 before", img2);
	// SIFT - 检测关键点并在原图中绘制
	int kp_number{ 1800 };
	vector<cv::KeyPoint> kp1, kp2;
	cv::Ptr<SiftFeatureDetector> siftdtc = SiftFeatureDetector::create(1800);
	siftdtc->detect(img1, kp1);
	vector<cv::KeyPoint>::iterator itvc;
	for (itvc = kp1.begin(); itvc != kp1.end(); itvc++)
	{
		
		itvc->angle = 10;
		/*itvc->octave = 0;
		itvc->size = 5;*/
		//cout << "angle:" << itvc->angle << "\t" << itvc->class_id << "\t" << itvc->octave << "\t" << "pt ->" << itvc->pt << "\t" << itvc->response << "\t" << itvc->size << endl;
	}
	Mat outimg1;
	cv::drawKeypoints(img1, kp1, outimg1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow("image1 keypoints", outimg1);
	siftdtc->detect(img2, kp2);
	vector<cv::KeyPoint>::iterator itvc1;
	for (itvc1 = kp2.begin(); itvc1 != kp2.end(); itvc1++)
	{
		
		itvc1->angle = 10;
		/*itvc1->octave = 0;
		itvc1->size = 5;*/
		//cout << "angle:" << itvc->angle << "\t" << itvc->class_id << "\t" << itvc->octave << "\t" << "pt ->" << itvc->pt << "\t" << itvc->response << "\t" << itvc->size << endl;
	}
	Mat outimg2;
	cv::drawKeypoints(img2, kp2, outimg2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow("image2 keypoints", outimg2);
	cout << "图1特征点数量：" << kp1.size() << endl;
	cout << "图2特征点数量：" << kp2.size() << endl;
	// SIFT - 特征向量提取
	cv::Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create();
	Mat descriptor1, descriptor2;
	extractor->compute(img1, kp1, descriptor1);
	extractor->compute(img2, kp2, descriptor2);
	//cv::imshow("desc", descriptor1);
	//cout << endl << "The size of feature matrix is: " << descriptor1.rows << "×" << descriptor1.cols << endl;
	// 两张图像的特征匹配
	/*cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
	vector<cv::DMatch> matches;
	Mat img_matches;
	matcher->match(descriptor1, descriptor2, matches);
	cv::drawMatches(img1, kp1, img2, kp2, matches, img_matches);
	imshow("matches", img_matches);*/
	BFMatcher bfm(cv::NORM_L2, true);
	vector<DMatch> dmatch1;
	bfm.match(descriptor1, descriptor2, dmatch1);
	Mat resbfmMap;
	drawMatches(img1, kp1, img2, kp2, dmatch1, resbfmMap);
	imshow("交叉匹配结果", resbfmMap);
	cout << "交叉匹配后：" << dmatch1.size() << endl;
	//imshow("bfmMatch",resbfmMap);
     vector<KeyPoint> Ransac_keypoint1, Ransac_keypoint2;
	for (int i = 0; i < dmatch1.size(); ++i) {
		Ransac_keypoint1.push_back(kp1[dmatch1[i].queryIdx]);
		Ransac_keypoint2.push_back(kp2[dmatch1[i].trainIdx]);
	}
	vector<Point2f> p01, p02;//float类型的特征点坐标
	for (int i = 0; i < dmatch1.size(); ++i) {
		p01.push_back(Ransac_keypoint1[i].pt);
		p02.push_back(Ransac_keypoint2[i].pt);
	}
	//利⽤基础矩阵提出误匹配点
	vector<uchar> RansacStatus;
	Mat Fundamental = findFundamentalMat(p01, p02, RansacStatus, FM_RANSAC);
	vector<KeyPoint> RR_keypoint1, RR_keypoint2;
	vector<DMatch> RR_matches;
	int index = 0;
	for (int i = 0; i < dmatch1.size(); ++i) {
		if (RansacStatus[i] != 0) {
			RR_keypoint1.push_back(Ransac_keypoint1[i]);
			RR_keypoint2.push_back(Ransac_keypoint2[i]);
			dmatch1[i].queryIdx = index;
			dmatch1[i].trainIdx = index;
			RR_matches.push_back(dmatch1[i]);
			index++;
		}
		cout << "内点数量：" << index << endl;
		Mat image_RR_matches;
		drawMatches(img1, RR_keypoint1, img2, RR_keypoint2, RR_matches, image_RR_matches);
		imshow("xiaochu", image_RR_matches); 
		//imwrite("/Users/houzhuolu/Desktop/res1.JPEG",image_RR_matches);
		waitKey(0);
		return 0;
	}
}