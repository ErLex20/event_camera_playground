#include "evo/lk_se3.hpp"

#include <glog/logging.h>

#include <random>

#include "evo/weight_functions.hpp"
#include "evo_utils/interpolation.hpp"

#define DEBUG_PRINT_LIMITS

//#define USE_WEIGHTED_UPDATE
//#define BLUR_EVENT_IMAGE

namespace evo {

void LKSE3::projectMap() {
    static std::vector<float> z_values;
    z_values.clear();

    map_local_->clear();

    cv::Size s = c_ref_.fullResolution();
    cv::Mat &depthmap = depth_ref_;
    depthmap = cv::Mat(s, CV_32F, cv::Scalar(0.));

    cv::Mat img(s, CV_32F, cv::Scalar(0.));

    n_visible_ = 0;
    size_t n_points = 0;

    Eigen::Affine3f T_ref_world = (T_world_kf_ * T_kf_ref_).inverse();

    for (const auto &P : map_->points) {
        Eigen::Vector3f p(P.x, P.y, P.z);
        p = T_ref_world * p;
        p[0] = p[0] / p[2] * c_ref_.fx() + c_ref_.cx();
        p[1] = p[1] / p[2] * c_ref_.fy() + c_ref_.cy();

        z_values.push_back(p[2]);
        ++n_points;

        if (p[0] < 0 || p[1] < 0) continue;

        int x = p[0] + .5f, y = p[1] + .5f;

        if (x >= s.width || y >= s.height) continue;

        float z = p[2];

        // Minimalistic occlusion detection
        float &depth = depthmap.at<float>(y, x);
        if (depth == 0.)
            depth = z;
        else if (depth > z)
            depth = z;

        img.at<float>(y, x) = 1.;

        map_local_->push_back(P);
        ++n_visible_;
    }

    const int k = map_blur_;
    cv::GaussianBlur(img, img, cv::Size(k, k), 0.);
    cv::GaussianBlur(depthmap, depthmap, cv::Size(k, k), 0.);
    depthmap /= img;
    ref_img_ = img;

    // Compute Median Depth
    std::nth_element(z_values.begin(), z_values.begin() + z_values.size() / 2,
                     z_values.end());
    depth_median_ = z_values[z_values.size() / 2];

    kf_visibility_ = static_cast<float>(n_visible_) / n_points;

    precomputeReferenceFrame();
}

void LKSE3::precomputeReferenceFrame() {
    cv::Mat grad_x_img, grad_y_img;
    cv::Sobel(ref_img_, grad_x_img, CV_32F, 1, 0);
    cv::Sobel(ref_img_, grad_y_img, CV_32F, 0, 1);

    // Image Pointers
    float *depth_ref = depth_ref_.ptr<float>(0),
          *grad_x = grad_x_img.ptr<float>(0),
          *grad_y = grad_y_img.ptr<float>(0);

    int w = ref_img_.cols, h = ref_img_.rows;

    keypoints_.clear();

    Vector8 vec = Vector8::Zero();
    Eigen::Map<const Vector6> vec6(&vec(0));

    for (size_t y = 0; y != static_cast<size_t>(h); ++y) {
        float v = ((float)y - c_ref_.cy()) / c_ref_.fy();

        for (size_t x = 0; x != static_cast<size_t>(w); ++x) {
            size_t offset = y * w + x;
            float z = depth_ref[offset];
            float pixel_value = ref_img_.at<float>(y, x);

            if (pixel_value < .01) continue;

            float u = ((float)x - c_ref_.cx()) / c_ref_.fx();

            float gx = grad_x[offset] * c_ref_.fx(),
                  gy = grad_y[offset] * c_ref_.fy();

            Vector8 v1, v2;
            v1 << -1. / z, 0., u / z, u * v, -(1. + u * u), v, 0., 0.;
            v2 << 0., -1. / z, v / z, 1 + v * v, -u * v, -u, 0., 0.;

            vec = gx * v1 + gy * v2;

            keypoints_.push_back(Keypoint(Eigen::Vector3f(u * z, v * z, z),
                                          pixel_value, vec,
                                          vec6 * vec6.transpose()));
        }
    }

    // Stochastic Gradient Descent
    static std::seed_seq seq{1, 2, 3, 4, 5};
    static std::mt19937 g(seq);
    std::shuffle(keypoints_.begin(), keypoints_.end(), g);

    // NOTE: integer division floors here (the original std::ceil was a no-op),
    // which keeps full-batch processing OOB-safe. Guard against 0 batches when
    // the reference frame has fewer than batch_size_ keypoints (e.g. a sparse
    // first keyframe), otherwise `iter % batches_` divides by zero (SIGFPE).
    batches_ = std::max<size_t>(1, keypoints_.size() / batch_size_);
}

void LKSE3::updateTransformation(const int offset, const int N,
                                 size_t pyr_lvl) {
    static Eigen::MatrixXf H;
    static Eigen::VectorXf Jres, dx;

    const cv::Mat &img = pyr_new_[pyr_lvl];
    const float *new_img = img.ptr<float>(0);
    float scale = std::pow(2.f, (float)pyr_lvl);
    float fx = fx_ / scale, fy = fy_ / scale, cx = cx_ / scale,
          cy = cy_ / scale;
    size_t w = img.cols, h = img.rows;

    H = Matrix6::Zero();
    Jres = Vector8::Zero();

    // Clamp the batch end to the available keypoints. For full batches
    // (keypoints_.size() >= batch_size_) this is a no-op and behaviour is
    // identical to the original; it only guards the degenerate last/partial
    // batch when the reference frame is sparse, preventing an OOB read.
    const int end =
        std::min<int>(offset + N, static_cast<int>(keypoints_.size()));
    for (auto i = offset; i != end; ++i) {
        const Keypoint &k = keypoints_[i];

        Eigen::Vector3f p = T_cur_ref_ * k.P;
        float u = p[0] / p[2] * fx + cx, v = p[1] / p[2] * fy + cy;

        float I_new = evo_utils::interpolate::bilinear(new_img, w, h, u, v);

        if (I_new == -1.f) continue;

        float res = I_new - k.pixel_value;

        if (res >= .95f) continue;

        Jres.noalias() += k.J * res;
        H.noalias() += k.JJt;
    }

    dx = H.ldlt().solve(Jres.head<6>() * scale);

    if ((bool)std::isnan((float)dx[0])) {
        LOG(WARNING) << "Matrix close to singular!";
        return;
    }

#ifdef USE_WEIGHTED_UPDATE
    for (size_t i = 0; i != 3; ++i)
        dx[i] *= weight_functions::Tukey(dx[i] * weight_scale_trans_ /
                                         depth_median_);
    for (size_t i = 3; i != 6; ++i)
        dx[i] *= weight_functions::Tukey(dx[i] * weight_scale_rot_);
#endif

    T_cur_ref_ *= SE3::exp(dx).matrix();
    x_ += dx;
}

void LKSE3::trackFrame() {
    T_cur_ref_ = T_ref_cam_.inverse();
    x_.setZero();

    for (size_t lvl = pyramid_levels_; lvl != 0; --lvl) {
        for (size_t iter = 0; iter != max_iterations_; ++iter) {
            updateTransformation((iter % batches_) * batch_size_, batch_size_,
                                 lvl - 1);
        }
    }
}

void LKSE3::drawEvents(EventQueue::iterator ev_first,
                       EventQueue::iterator ev_last, cv::Mat &out) {
    // Precompute rectification table
    static std::vector<cv::Point> points;
    static std::vector<Eigen::Vector4f> weights;
    if (points.size() == 0) {
        cv::Rect rect(0, 0, width_ - 1, height_ - 1);

        for (size_t y = 0; y != height_; ++y)
            for (size_t x = 0; x != width_; ++x) {
                cv::Point2d p = c_.rectifyPoint(cv::Point(x, y));
                cv::Point tl(std::floor(p.x), std::floor(p.y));
                Eigen::Vector4f w(0, 0, 0, 0);

                if (rect.contains(tl)) {
                    const float fx = p.x - tl.x, fy = p.y - tl.y;

                    w[0] = (1.f - fx) * (1.f - fy);
                    w[1] = (fx) * (1.f - fy);
                    w[2] = (1.f - fx) * (fy);
                    w[3] = (fx) * (fy);
                } else {
                    tl.x = -1;
                }

                points.push_back(tl);
                weights.push_back(w);
            }
    }

    // Draw Events
    auto draw = [](float &p, const float val) { p = std::min(p + val, 1.f); };

    out = cv::Scalar(0);

    for (auto e = ev_first; e != ev_last; ++e) {
        const cv::Point &p = points[e->x + e->y * width_];
        if (p.x == -1) continue;

        const Eigen::Vector4f &w = weights[e->x + e->y * width_];
        draw(out.at<float>(p.y, p.x), w[0]);
        draw(out.at<float>(p.y, p.x + 1), w[1]);
        draw(out.at<float>(p.y + 1, p.x), w[2]);
        draw(out.at<float>(p.y + 1, p.x + 1), w[3]);
    }
}

void LKSE3::drawEventsNN(EventQueue::iterator ev_first,
                         EventQueue::iterator ev_last, cv::Mat &out) {
    // Precompute rectification table
    static std::vector<cv::Point> points;
    if (points.size() == 0) {
        for (size_t y = 0; y != height_; ++y)
            for (size_t x = 0; x != width_; ++x) {
                cv::Point p = c_.rectifyPoint(cv::Point(x, y));

                if (!rect_.contains(p)) p.x = -1;

                points.push_back(p);
            }
    }

    out = cv::Scalar(0);

    for (auto e = ev_first; e != ev_last; ++e) {
        const cv::Point &p = points[e->x + e->y * width_];
        if (p.x == -1) continue;

        out.at<float>(p) = 1.;
    }

#ifdef BLUR_EVENT_IMAGE
    const int k = map_blur_;
    cv::GaussianBlur(out, out, cv::Size(k, k), 0.);
#endif
}

}  // namespace evo
