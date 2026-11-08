#pragma once
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

namespace hd {

template <typename T>
HD inline T max_of(T a, T b) { return (a < b) ? b : a; }

template <typename T>
HD inline T min_of(T a, T b) { return (b < a) ? b : a; }

HD inline double abs_of(double v) {
#ifdef __CUDA_ARCH__
    return fabs(v);
#else
    return std::fabs(v);
#endif
}

HD inline double sqrt_of(double v) {
#ifdef __CUDA_ARCH__
    return sqrt(v);
#else
    return std::sqrt(v);
#endif
}

HD inline double hypot_of(double x, double y) {
#ifdef __CUDA_ARCH__
    return hypot(x, y);
#else
    return std::hypot(x, y);
#endif
}

HD inline bool is_finite(double v) {
#ifdef __CUDA_ARCH__
    return isfinite(v);
#else
    return std::isfinite(v);
#endif
}

HD inline int hamming256(const uint8_t* a, const uint8_t* b) {
    int dist = 0;
#ifdef __CUDA_ARCH__
    for (int i = 0; i < 32; i += 4) {
        uint32_t x = static_cast<uint32_t>(a[i] ^ b[i]) |
                     (static_cast<uint32_t>(a[i + 1] ^ b[i + 1]) << 8) |
                     (static_cast<uint32_t>(a[i + 2] ^ b[i + 2]) << 16) |
                     (static_cast<uint32_t>(a[i + 3] ^ b[i + 3]) << 24);
        dist += __popc(x);
    }
#else
    for (int i = 0; i < 32; ++i) {
        uint8_t x = a[i] ^ b[i];
        while (x) {
            dist++;
            x &= (x - 1);
        }
    }
#endif
    return dist;
}

HD inline bool solve_linear_8(double A[8][8], double b[8], double x[8]) {
    for (int col = 0; col < 8; ++col) {
        int pivot = col;
        double max_val = abs_of(A[col][col]);
        for (int row = col + 1; row < 8; ++row) {
            if (abs_of(A[row][col]) > max_val) {
                max_val = abs_of(A[row][col]);
                pivot = row;
            }
        }
        if (max_val < 1e-12) return false;

        if (pivot != col) {
            for (int k = col; k < 8; ++k) {
                double t = A[col][k];
                A[col][k] = A[pivot][k];
                A[pivot][k] = t;
            }
            double t = b[col];
            b[col] = b[pivot];
            b[pivot] = t;
        }

        double diag = A[col][col];
        for (int k = col; k < 8; ++k) A[col][k] /= diag;
        b[col] /= diag;

        for (int row = 0; row < 8; ++row) {
            if (row != col) {
                double factor = A[row][col];
                for (int k = col; k < 8; ++k) A[row][k] -= factor * A[col][k];
                b[row] -= factor * b[col];
            }
        }
    }
    for (int i = 0; i < 8; ++i) x[i] = b[i];
    return true;
}

HD inline void multiply_3x3(const double a[9], const double b[9], double out[9]) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out[3*r+c] = 0.0;
            for (int k = 0; k < 3; ++k) out[3*r+c] += a[3*r+k] * b[3*k+c];
        }
    }
}

struct DltNorm {
    double csx, csy, cdx, cdy, ss, sd;
};

HD inline bool dlt_finish_norm(double src_spread, double dst_spread, double n, DltNorm& m) {
    double src_mean = src_spread / n;
    double dst_mean = dst_spread / n;
    if (src_mean < 1e-9 || dst_mean < 1e-9) return false;
    m.ss = sqrt_of(2.0) / src_mean;
    m.sd = sqrt_of(2.0) / dst_mean;
    return true;
}

HD inline void dlt_accumulate(const DltNorm& m, float sx, float sy, float dx, float dy,
                              double ata[8][8], double atb[8]) {
    double xs = m.ss * (sx - m.csx), ys = m.ss * (sy - m.csy);
    double xd = m.sd * (dx - m.cdx), yd = m.sd * (dy - m.cdy);
    double rows[2][8] = {
        {xs, ys, 1, 0, 0, 0, -xd*xs, -xd*ys},
        {0, 0, 0, xs, ys, 1, -yd*xs, -yd*ys}
    };
    double rhs[2] = {xd, yd};
    for (int q = 0; q < 2; ++q) {
        for (int r = 0; r < 8; ++r) {
            atb[r] += rows[q][r] * rhs[q];
            for (int c = 0; c < 8; ++c) ata[r][c] += rows[q][r] * rows[q][c];
        }
    }
}

HD inline bool dlt_solve(double ata[8][8], double atb[8], const DltNorm& m, double out[9]) {
    double x[8];
    if (!solve_linear_8(ata, atb, x)) return false;
    double hn[9] = {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1.0};
    double ts[9] = {m.ss, 0, -m.ss*m.csx, 0, m.ss, -m.ss*m.csy, 0, 0, 1};
    double td_inv[9] = {1.0/m.sd, 0, m.cdx, 0, 1.0/m.sd, m.cdy, 0, 0, 1};
    double temp[9], h[9];
    multiply_3x3(hn, ts, temp);
    multiply_3x3(td_inv, temp, h);
    if (!is_finite(h[8]) || abs_of(h[8]) < 1e-12) return false;
    for (int i = 0; i < 9; ++i) out[i] = h[i] / h[8];
    return true;
}

HD inline bool dlt_fit(const float* src_xy, const float* dst_xy, int n, double out[9]) {
    if (n < 4) return false;
    DltNorm m = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < n; ++i) {
        m.csx += src_xy[2*i]; m.csy += src_xy[2*i+1];
        m.cdx += dst_xy[2*i]; m.cdy += dst_xy[2*i+1];
    }
    double count = static_cast<double>(n);
    m.csx /= count; m.csy /= count; m.cdx /= count; m.cdy /= count;
    double src_spread = 0, dst_spread = 0;
    for (int i = 0; i < n; ++i) {
        src_spread += hypot_of(src_xy[2*i] - m.csx, src_xy[2*i+1] - m.csy);
        dst_spread += hypot_of(dst_xy[2*i] - m.cdx, dst_xy[2*i+1] - m.cdy);
    }
    if (!dlt_finish_norm(src_spread, dst_spread, count, m)) return false;
    double ata[8][8] = {};
    double atb[8] = {};
    for (int i = 0; i < n; ++i)
        dlt_accumulate(m, src_xy[2*i], src_xy[2*i+1], dst_xy[2*i], dst_xy[2*i+1], ata, atb);
    return dlt_solve(ata, atb, m, out);
}

HD inline bool invert_h(const double m[9], double inv[9]) {
    double det = m[0] * (m[4] * m[8] - m[5] * m[7]) -
                 m[1] * (m[3] * m[8] - m[5] * m[6]) +
                 m[2] * (m[3] * m[7] - m[4] * m[6]);

    if (abs_of(det) < 1e-12) return false;
    double inv_det = 1.0 / det;

    inv[0] =  (m[4] * m[8] - m[5] * m[7]) * inv_det;
    inv[1] = -(m[1] * m[8] - m[2] * m[7]) * inv_det;
    inv[2] =  (m[1] * m[5] - m[2] * m[4]) * inv_det;
    inv[3] = -(m[3] * m[8] - m[5] * m[6]) * inv_det;
    inv[4] =  (m[0] * m[8] - m[2] * m[6]) * inv_det;
    inv[5] = -(m[0] * m[5] - m[2] * m[3]) * inv_det;
    inv[6] =  (m[3] * m[7] - m[4] * m[6]) * inv_det;
    inv[7] = -(m[0] * m[7] - m[1] * m[6]) * inv_det;
    inv[8] =  (m[0] * m[4] - m[1] * m[3]) * inv_det;
    return true;
}

HD inline void project_point(const double h[9], float x, float y, float& ox, float& oy) {
    double wx = h[0] * x + h[1] * y + h[2];
    double wy = h[3] * x + h[4] * y + h[5];
    double wz = h[6] * x + h[7] * y + h[8];
    if (abs_of(wz) < 1e-12) wz = 1e-12;
    ox = static_cast<float>(wx / wz);
    oy = static_cast<float>(wy / wz);
}

HD inline double symmetric_error_sq(const double h[9], const double hinv[9],
                                    float ax, float ay, float bx, float by) {
    float pax, pay, pbx, pby;
    project_point(h, bx, by, pax, pay);
    project_point(hinv, ax, ay, pbx, pby);
    double fdx = pax - ax, fdy = pay - ay;
    double bdx = pbx - bx, bdy = pby - by;
    return 0.5 * (fdx*fdx + fdy*fdy + bdx*bdx + bdy*bdy);
}

HD inline bool sample_bilinear(const uint8_t* rgb, int width, int height,
                               float x, float y, float out_rgb[3]) {
    if (x < 0.0f || x >= width - 1 || y < 0.0f || y >= height - 1) return false;

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float dx = x - x0;
    float dy = y - y0;
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w10 = dx * (1.0f - dy);
    float w01 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    for (int c = 0; c < 3; ++c) {
        float p00 = rgb[(y0 * width + x0) * 3 + c];
        float p10 = rgb[(y0 * width + x1) * 3 + c];
        float p01 = rgb[(y1 * width + x0) * 3 + c];
        float p11 = rgb[(y1 * width + x1) * 3 + c];
        out_rgb[c] = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11;
    }
    return true;
}

}
