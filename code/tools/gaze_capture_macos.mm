#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#include "../device/tracking/eye_tracking_optimized.h"

@interface GazeWsClient : NSObject
@property(nonatomic, strong) NSURLSessionWebSocketTask *task;
- (instancetype)initWithURL:(NSURL *)url;
- (void)sendJSON:(const std::string &)json;
@end

@implementation GazeWsClient
- (instancetype)initWithURL:(NSURL *)url {
    self = [super init];
    if (self) {
        NSURLSession *session = [NSURLSession sessionWithConfiguration:[NSURLSessionConfiguration defaultSessionConfiguration]];
        _task = [session webSocketTaskWithURL:url];
        [_task resume];
    }
    return self;
}

- (void)sendJSON:(const std::string &)json {
    NSString *payload = [NSString stringWithUTF8String:json.c_str()];
    NSURLSessionWebSocketMessage *message = [[NSURLSessionWebSocketMessage alloc] initWithString:payload];
    [_task sendMessage:message completionHandler:^(NSError * _Nullable error) {
        if (error) {
            NSLog(@"WebSocket send error: %@", error);
        }
    }];
}
@end

static bool DecodeJPEGToRGB(NSString *b64, std::vector<uint8_t> &outRgb, int &outW, int &outH) {
    if (!b64 || b64.length == 0) return false;
    NSData *jpeg = [[NSData alloc] initWithBase64EncodedString:b64 options:0];
    if (!jpeg || jpeg.length == 0) return false;

    CGImageSourceRef src = CGImageSourceCreateWithData((__bridge CFDataRef)jpeg, NULL);
    if (!src) return false;
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, NULL);
    CFRelease(src);
    if (!img) return false;

    const size_t width = CGImageGetWidth(img);
    const size_t height = CGImageGetHeight(img);
    const size_t bytesPerRow = width * 4;
    std::vector<uint8_t> rgba(width * height * 4);

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        rgba.data(),
        width,
        height,
        8,
        bytesPerRow,
        colorSpace,
        kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst
    );
    CGColorSpaceRelease(colorSpace);
    if (!ctx) {
        CGImageRelease(img);
        return false;
    }

    CGContextDrawImage(ctx, CGRectMake(0, 0, width, height), img);
    CGContextRelease(ctx);
    CGImageRelease(img);

    outRgb.assign(width * height * 3, 0);
    for (size_t i = 0; i < width * height; ++i) {
        const uint8_t b = rgba[i * 4 + 0];
        const uint8_t g = rgba[i * 4 + 1];
        const uint8_t r = rgba[i * 4 + 2];
        outRgb[i * 3 + 0] = r;
        outRgb[i * 3 + 1] = g;
        outRgb[i * 3 + 2] = b;
    }

    outW = (int)width;
    outH = (int)height;
    return true;
}

@interface GazeCapture : NSObject<AVCaptureVideoDataOutputSampleBufferDelegate>
@property(nonatomic, strong) AVCaptureSession *session;
@property(nonatomic, strong) GazeWsClient *ws;
@end

@implementation GazeCapture {
    EyeTrackingSystem tracker_;
    std::vector<uint8_t> rgb_;
    std::chrono::steady_clock::time_point lastFrame_;
    std::mutex lock_;
}

- (instancetype)initWithWS:(GazeWsClient *)ws {
    self = [super init];
    if (self) {
        _ws = ws;
        lastFrame_ = std::chrono::steady_clock::now();
        tracker_.initialize();
    }
    return self;
}

- (void)start {
    self.session = [[AVCaptureSession alloc] init];
    [self.session beginConfiguration];
    self.session.sessionPreset = AVCaptureSessionPreset640x480;

    AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    NSError *error = nil;
    AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
    if (!input) {
        NSLog(@"Camera input error: %@", error);
        return;
    }
    [self.session addInput:input];

    AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
    output.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)};
    dispatch_queue_t queue = dispatch_queue_create("gaze.capture.queue", DISPATCH_QUEUE_SERIAL);
    [output setSampleBufferDelegate:self queue:queue];
    output.alwaysDiscardsLateVideoFrames = YES;
    [self.session addOutput:output];

    [self.session commitConfiguration];
    [self.session startRunning];
}

- (void)captureOutput:(AVCaptureOutput *)output
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFrame_).count();
    if (elapsed < 1000) {
        return;
    }
    lastFrame_ = now;

    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (!imageBuffer) return;

    CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
    const size_t width = CVPixelBufferGetWidth(imageBuffer);
    const size_t height = CVPixelBufferGetHeight(imageBuffer);
    const size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    const uint8_t *base = (const uint8_t *)CVPixelBufferGetBaseAddress(imageBuffer);
    if (!base) {
        CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
        return;
    }

    const size_t outSize = width * height * 3;
    if (rgb_.size() != outSize) {
        rgb_.assign(outSize, 0);
    }

    for (size_t y = 0; y < height; ++y) {
        const uint8_t *row = base + y * bytesPerRow;
        uint8_t *dst = rgb_.data() + y * width * 3;
        for (size_t x = 0; x < width; ++x) {
            const uint8_t b = row[x * 4 + 0];
            const uint8_t g = row[x * 4 + 1];
            const uint8_t r = row[x * 4 + 2];
            dst[x * 3 + 0] = r;
            dst[x * 3 + 1] = g;
            dst[x * 3 + 2] = b;
        }
    }

    CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

    bool detected = false;
    {
        std::lock_guard<std::mutex> guard(lock_);
        detected = tracker_.process_frame(rgb_.data(), (int)width, (int)height);
    }

    if (detected) {
        float gaze_x = 0.0f, gaze_y = 0.0f;
        tracker_.get_gaze(gaze_x, gaze_y);
        const bool blink = tracker_.is_blinking();

        char buf[256];
        snprintf(
            buf,
            sizeof(buf),
            "{\"type\":\"gaze\",\"client_id\":\"default\",\"gaze\":{\"x\":%.3f,\"y\":%.3f,\"blink\":%s}}",
            gaze_x,
            gaze_y,
            blink ? "true" : "false"
        );
        [_ws sendJSON:std::string(buf)];
    }
}
@end

@interface GazeFrameWorker : NSObject
@property(nonatomic, strong) NSURLSessionWebSocketTask *task;
- (instancetype)initWithURL:(NSURL *)url;
- (void)start;
@end

@implementation GazeFrameWorker {
    EyeTrackingSystem tracker_;
    std::vector<uint8_t> rgb_;
    std::mutex lock_;
    uint32_t frame_count_;
}

- (instancetype)initWithURL:(NSURL *)url {
    self = [super init];
    if (self) {
        NSURLSession *session = [NSURLSession sessionWithConfiguration:[NSURLSessionConfiguration defaultSessionConfiguration]];
        _task = [session webSocketTaskWithURL:url];
        [_task resume];
        frame_count_ = 0;
        fprintf(stderr, "[GazeWorker] WS connect: %s\n", [[url absoluteString] UTF8String]);
        fflush(stderr);
        tracker_.initialize();
    }
    return self;
}

- (void)sendJSON:(NSDictionary *)obj {
    NSError *err = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:obj options:0 error:&err];
    if (!data) return;
    NSString *payload = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    NSURLSessionWebSocketMessage *message = [[NSURLSessionWebSocketMessage alloc] initWithString:payload];
    [_task sendMessage:message completionHandler:^(NSError * _Nullable error) {
        if (error) {
            NSLog(@"WebSocket send error: %@", error);
        }
    }];
}

- (void)handleFrame:(NSDictionary *)frame {
    NSString *b64 = frame[@"image_b64"];
    if (!b64) {
        fprintf(stderr, "[GazeWorker] frame missing image_b64\n");
        fflush(stderr);
        return;
    }

    int w = 0, h = 0;
    if (!DecodeJPEGToRGB(b64, rgb_, w, h)) {
        fprintf(stderr, "[GazeWorker] JPEG decode failed\n");
        fflush(stderr);
        return;
    }
    frame_count_++;
    if (frame_count_ % 30 == 0) {
        fprintf(stderr, "[GazeWorker] frames=%u w=%d h=%d\n", frame_count_, w, h);
        fflush(stderr);
    }

    bool detected = false;
    {
        std::lock_guard<std::mutex> guard(lock_);
        detected = tracker_.process_frame(rgb_.data(), w, h);
    }

    if (detected) {
        float gaze_x = 0.0f, gaze_y = 0.0f;
        tracker_.get_gaze(gaze_x, gaze_y);
        const bool blink = tracker_.is_blinking();
        NSDictionary *payload = @{
            @"type": @"gaze",
            @"client_id": @"default",
            @"gaze": @{
                @"x": @(gaze_x),
                @"y": @(gaze_y),
                @"blink": @(blink)
            }
        };
        [self sendJSON:payload];
    }
    if (frame_count_ % 30 == 0) {
        fprintf(stderr, "[GazeWorker] detected=%s\n", detected ? "true" : "false");
        fflush(stderr);
    }
}

- (void)receiveLoop {
    [_task receiveMessageWithCompletionHandler:^(NSURLSessionWebSocketMessage * _Nullable message, NSError * _Nullable error) {
        if (error) {
            NSLog(@"WebSocket receive error: %@", error);
            return;
        }
        if (!message) return;

        NSString *text = message.string;
        if (text) {
            NSData *data = [text dataUsingEncoding:NSUTF8StringEncoding];
            if (data) {
                NSError *err = nil;
                NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
                if ([json isKindOfClass:[NSDictionary class]]) {
                    NSString *type = json[@"type"];
                    if ([type isEqualToString:@"gaze_frame"]) {
                        [self handleFrame:json];
                    }
                }
            }
        }
        [self receiveLoop];
    }];
}

- (void)start {
    NSURLSessionWebSocketMessage *hello = [[NSURLSessionWebSocketMessage alloc] initWithString:@"{\"type\":\"gaze_hello\"}"];
    [_task sendMessage:hello completionHandler:^(NSError * _Nullable error) {
        if (error) {
            NSLog(@"WebSocket hello error: %@", error);
        } else {
            fprintf(stderr, "[GazeWorker] hello sent\n");
            fflush(stderr);
        }
    }];
    [self receiveLoop];
}
@end

int main(int argc, char **argv) {
    @autoreleasepool {
        setvbuf(stdout, NULL, _IONBF, 0);
        setvbuf(stderr, NULL, _IONBF, 0);
        const char *input = getenv("OCTO_GAZE_INPUT_WS");
        if (input && strlen(input) > 0) {
            NSURL *url = [NSURL URLWithString:[NSString stringWithUTF8String:input]];
            GazeFrameWorker *worker = [[GazeFrameWorker alloc] initWithURL:url];
            [worker start];
        } else {
            NSURL *url = [NSURL URLWithString:@"ws://localhost:8766"];
            GazeWsClient *ws = [[GazeWsClient alloc] initWithURL:url];
            GazeCapture *capture = [[GazeCapture alloc] initWithWS:ws];
            [capture start];
        }
        [[NSRunLoop currentRunLoop] run];
    }
    return 0;
}
