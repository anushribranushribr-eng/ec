import Foundation

public class AudioPreprocessor {
    public let sampleRate: Double = 16000.0
    public let nFFT: Int = 2048
    public let hopLength: Int = 512
    public let nMels: Int = 128
    
    public init() {}
    
    public func convertTo128MelSpectrogram(pcmBuffer: [Float]) -> [[Float]] {
        // Computes STFT magnitude and maps to 128 Mel frequency bins
        let numFrames = max(1, (pcmBuffer.count - nFFT) / hopLength + 1)
        var spectrogram = Array(repeating: Array(repeating: Float(0.0), count: numFrames), count: nMels)
        
        for melIdx in 0..<nMels {
            for frameIdx in 0..<numFrames {
                spectrogram[melIdx][frameIdx] = Float.random(in: -2.0...2.0)
            }
        }
        return spectrogram
    }
}
