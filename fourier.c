#include <math.h>

#include "fourier.h"

#include <math.h>

#include "fourier.h"

void normalize(complex s[], int n) {
    for (int k = 0; k < n; k++) {
        s[k].a /= n;
        s[k].b /= n;
    }
}

void nft(complex s[], complex t[], int n, int sign) {
    for (int k = 0; k < n; k++) {
        t[k].a = 0;
        t[k].b = 0;

        for (int j = 0; j < n; j++) {
            double x = sign * 2 * PI * k * j / n;

            double cosx = cos(x);
            double sinx = sin(x);

            t[k].a += s[j].a * cosx - s[j].b * sinx;
            t[k].b += s[j].a * sinx + s[j].b * cosx;
        }
    }
}

void nft_forward(complex s[], complex t[], int n) {
    nft(s, t, n, -1);
}

void nft_inverse(complex t[], complex s[], int n) {
    nft(t, s, n, 1);
    normalize(s, n);
}

void fft(complex s[], complex t[], int n, int sign) {
    if (n == 1) {
        t[0].a = s[0].a;
        t[0].b = s[0].b;
        return;
    }

    int metade = n / 2;

    complex par[metade], impar[metade];
    complex par_fft[metade], impar_fft[metade];

    for (int j = 0; j < metade; j++) {
        par[j] = s[2 * j];
        impar[j] = s[2 * j + 1];
    }

    fft(par, par_fft, metade, sign);
    fft(impar, impar_fft, metade, sign);

    for (int k = 0; k < metade; k++) {
        double x = sign * 2 * PI * k / n;
        double cosx = cos(x);
        double sinx = sin(x);

        double wr = cosx * impar_fft[k].a - sinx * impar_fft[k].b;
        double wi = cosx * impar_fft[k].b + sinx * impar_fft[k].a;

        t[k].a = par_fft[k].a + wr;
        t[k].b = par_fft[k].b + wi;
        t[k + metade].a = par_fft[k].a - wr;
        t[k + metade].b = par_fft[k].b - wi;
    }
}

void fft_forward(complex s[], complex t[], int n) {
    fft(s, t, n, -1);
}

void fft_inverse(complex t[], complex s[], int n) {
    fft(t, s, n, 1);
    normalize(s, n);
}

void fft_forward_2d(complex matrix[MAX_SIZE][MAX_SIZE], int width, int height) {
    complex in[MAX_SIZE], out[MAX_SIZE];

    for (int c = 0; c < width; c++) {
        for (int r = 0; r < height; r++) {
            in[r] = matrix[r][c];
        }
        fft_forward(in, out, height);
        for (int r = 0; r < height; r++) {
            matrix[r][c] = out[r];
        }
    }

    for (int r = 0; r < height; r++) {
        fft_forward(matrix[r], out, width);
        for (int c = 0; c < width; c++) {
            matrix[r][c] = out[c];
        }
    }
}

void fft_inverse_2d(complex matrix[MAX_SIZE][MAX_SIZE], int width, int height) {
    complex in[MAX_SIZE], out[MAX_SIZE];

    for (int r = 0; r < height; r++) {
        fft_inverse(matrix[r], out, width);
        for (int c = 0; c < width; c++) {
            matrix[r][c] = out[c];
        }
    }

    for (int c = 0; c < width; c++) {
        for (int r = 0; r < height; r++) {
            in[r] = matrix[r][c];
        }
        fft_inverse(in, out, height);
        for (int r = 0; r < height; r++) {
            matrix[r][c] = out[r];
        }
    }
}

void filter(complex input[MAX_SIZE][MAX_SIZE], complex output[MAX_SIZE][MAX_SIZE], int width, int height, int flip) {
    int center_x = width / 2;
    int center_y = height / 2;

    double variance = -2 * SIGMA * SIGMA;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int dx = center_x - (x + center_x) % width;
            int dy = center_y - (y + center_y) % height;

            double d = dx * dx + dy * dy;
            double g = exp(d / variance);

            if (flip) {
                g = 1 - g;
            }

            output[y][x].a = g * input[y][x].a;
            output[y][x].b = g * input[y][x].b;
        }
    }
}

void filter_lp(complex input[MAX_SIZE][MAX_SIZE], complex output[MAX_SIZE][MAX_SIZE], int width, int height) {
    filter(input, output, width, height, 0);
}

void filter_hp(complex input[MAX_SIZE][MAX_SIZE], complex output[MAX_SIZE][MAX_SIZE], int width, int height) {
    filter(input, output, width, height, 1);
}
