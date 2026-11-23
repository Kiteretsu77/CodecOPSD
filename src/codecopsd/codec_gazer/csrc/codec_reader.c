/* On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).
 *
 * Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng
 *
 * Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.
 *
 * Codec pruning is a contribution of this paper; please cite it if you use, modify, or port this code.
 */

/* Codec signals of an H.264 file decoded with the patched FFmpeg 5.1 (ffmpeg-5.1.patch), per frame and 16 x 16
 * macroblock: the largest squared L0 motion vector (whole pixels) of its 4 x 4 cells, the sum of |luma residual| (the
 * patched decoder outputs 128 + residual, 128 where nothing is coded) and whether it was skipped.
 * Built by scripts/build_codec_reader.sh. */
#define PY_SSIZE_T_CLEAN
#define Py_LIMITED_API 0x030A0000
#include <Python.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/motion_vector.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int frames, height, width, threads, decoded;
    int32_t *motion;  /* [frames, height / 16, width / 16] */
    int32_t *res;     /* [frames, height / 16, width / 16] */
    uint8_t *skip;    /* [frames, height / 16, width / 16] */
    int16_t *cells;   /* one frame's motion vectors on the 4 x 4 pixel grid: [height / 4, width / 4, 2] */
    char error[256];
} Signals;

static void store_frame(Signals *s, const AVFrame *frame)
{
    const int h4 = s->height / 4, w4 = s->width / 4, h16 = s->height / 16, w16 = s->width / 16;
    const size_t offset = (size_t)s->decoded * h16 * w16;
    int32_t *motion = s->motion + offset, *res = s->res + offset;
    uint8_t *skip = s->skip + offset;
    const AVFrameSideData *sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);

    for (int y = 0; y < s->height; y++) {
        const uint8_t *row = frame->data[0] + (size_t)y * frame->linesize[0];
        int32_t *mb = res + (y / 16) * w16;
        for (int x = 0; x < s->width; x++)
            mb[x / 16] += abs(row[x] - 128);
    }
    memset(s->cells, 0, (size_t)h4 * w4 * 2 * sizeof(int16_t));
    if (sd) {
        const AVMotionVector *mvs = (const AVMotionVector *)sd->data;
        for (size_t i = 0; i < sd->size / sizeof(*mvs); i++) {
            const AVMotionVector *m = &mvs[i];
            if ((m->flags & 1) && m->dst_y / 16 < h16 && m->dst_x / 16 < w16)
                skip[(m->dst_y / 16) * w16 + m->dst_x / 16] = 1;
            if (m->source > 0)  /* L1 */
                continue;
            /* the block centred at (dst_x, dst_y) covers w / 4 x h / 4 cells; later vectors overwrite earlier ones */
            for (int dx = -m->w / 8; dx < m->w / 8; dx++)
                for (int dy = -m->h / 8; dy < m->h / 8; dy++) {
                    const int x = m->dst_x / 4 + dx, y = m->dst_y / 4 + dy;
                    if (x >= 0 && x < w4 && y >= 0 && y < h4) {
                        s->cells[(y * w4 + x) * 2] = m->src_x - m->dst_x;
                        s->cells[(y * w4 + x) * 2 + 1] = m->src_y - m->dst_y;
                    }
                }
        }
    }
    for (int y = 0; y < h4; y++)
        for (int x = 0; x < w4; x++) {
            const int32_t vx = s->cells[(y * w4 + x) * 2], vy = s->cells[(y * w4 + x) * 2 + 1];
            int32_t *mb = motion + (y / 4) * w16 + x / 4;
            if (vx * vx + vy * vy > *mb)
                *mb = vx * vx + vy * vy;
        }
    s->decoded++;
}

static int decode(Signals *s, AVCodecContext *dec, const AVPacket *pkt, AVFrame *frame)
{
    int ret = avcodec_send_packet(dec, pkt);
    while (ret >= 0) {
        ret = avcodec_receive_frame(dec, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return 0;
        if (ret < 0)
            break;
        if (frame->width != s->width || frame->height != s->height) {
            snprintf(s->error, sizeof(s->error), "frame is %dx%d, expected %dx%d", frame->width, frame->height,
                     s->width, s->height);
            return -1;
        }
        if (s->decoded < s->frames)
            store_frame(s, frame);
        av_frame_unref(frame);
    }
    snprintf(s->error, sizeof(s->error), "decoding failed: %s", av_err2str(ret));
    return -1;
}

static int read_signals(const char *path, Signals *s)
{
    AVFormatContext *fmt = NULL;
    AVCodecContext *dec = NULL;
    const AVCodec *codec = NULL;
    AVDictionary *opts = NULL;
    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    int ret = -1, stream;

    if (avformat_open_input(&fmt, path, NULL, NULL) < 0 || avformat_find_stream_info(fmt, NULL) < 0 ||
        (stream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0)) < 0 ||
        codec->id != AV_CODEC_ID_H264) {
        snprintf(s->error, sizeof(s->error), "cannot open an H.264 video stream in %s", path);
        goto end;
    }
    dec = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(dec, fmt->streams[stream]->codecpar);
    dec->thread_count = s->threads;
    dec->thread_type = FF_THREAD_FRAME;
    av_dict_set(&opts, "flags2", "+export_mvs", 0);
    av_dict_set(&opts, "skip_loop_filter", "all", 0);  /* no deblocking: the residual stays as decoded */
    if (avcodec_open2(dec, codec, &opts) < 0) {
        snprintf(s->error, sizeof(s->error), "cannot open the H.264 decoder");
        goto end;
    }
    while (av_read_frame(fmt, pkt) >= 0) {
        const int failed = pkt->stream_index == stream && decode(s, dec, pkt, frame) < 0;
        av_packet_unref(pkt);
        if (failed)
            goto end;
    }
    if (decode(s, dec, NULL, frame) < 0)
        goto end;
    if (s->decoded != s->frames) {
        snprintf(s->error, sizeof(s->error), "decoded %d frames, expected %d", s->decoded, s->frames);
        goto end;
    }
    ret = 0;
end:
    av_dict_free(&opts);
    avcodec_free_context(&dec);
    avformat_close_input(&fmt);
    av_packet_free(&pkt);
    av_frame_free(&frame);
    return ret;
}

static PyObject *py_read(PyObject *self, PyObject *args)
{
    const char *path;
    Signals s = {0};
    if (!PyArg_ParseTuple(args, "siiii", &path, &s.frames, &s.height, &s.width, &s.threads))
        return NULL;
    if (s.frames < 1 || s.height % 16 || s.width % 16 || s.height < 16 || s.width < 16) {
        PyErr_SetString(PyExc_ValueError, "need frames >= 1 and height, width multiples of 16");
        return NULL;
    }
    const Py_ssize_t n = (Py_ssize_t)s.frames * (s.height / 16) * (s.width / 16);
    PyObject *motion = PyBytes_FromStringAndSize(NULL, n * 4), *res = PyBytes_FromStringAndSize(NULL, n * 4);
    PyObject *skip = PyBytes_FromStringAndSize(NULL, n);
    s.cells = malloc((size_t)(s.height / 4) * (s.width / 4) * 2 * sizeof(int16_t));
    if (!motion || !res || !skip || !s.cells)
        goto fail;
    s.motion = (int32_t *)PyBytes_AsString(motion);
    s.res = (int32_t *)PyBytes_AsString(res);
    s.skip = (uint8_t *)PyBytes_AsString(skip);
    memset(s.motion, 0, n * 4);
    memset(s.res, 0, n * 4);
    memset(s.skip, 0, n);
    int ret;
    Py_BEGIN_ALLOW_THREADS
    ret = read_signals(path, &s);
    Py_END_ALLOW_THREADS
    if (ret < 0) {
        PyErr_SetString(PyExc_RuntimeError, s.error);
        goto fail;
    }
    free(s.cells);
    return Py_BuildValue("(NNN)", motion, res, skip);
fail:
    free(s.cells);
    Py_XDECREF(motion);
    Py_XDECREF(res);
    Py_XDECREF(skip);
    return NULL;
}

static PyMethodDef methods[] = {
    {"read", py_read, METH_VARARGS,
     "read(path, frames, height, width, threads) -> (largest squared motion vector int32, sum of |luma residual| "
     "int32, skipped uint8), each [frames, height / 16, width / 16] as bytes."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_codec_reader", NULL, -1, methods};

PyMODINIT_FUNC PyInit__codec_reader(void)
{
    av_log_set_level(AV_LOG_ERROR);
    return PyModule_Create(&module);
}
