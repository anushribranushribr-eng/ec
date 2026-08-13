import Foundation
import AVFoundation

public class AudioManager: ObservableObject {
    @Published public var isListening: Bool = false
    @Published public var currentAudioLevel: Float = 0.0
    
    private var audioEngine: AVAudioEngine?
    private var preprocessor = AudioPreprocessor()
    public var onAudioChunkProcessed: (([[Float]]) -> Void)?
    
    public init() {}
    
    public func startListening() {
        isListening = true
        // Configure AVAudioSession for 16,000 Hz Mono input
        #if os(iOS)
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: .duckOthers)
            try session.setActive(true)
        } catch {
            print("Failed to configure AVAudioSession: \(error)")
        }
        #endif
        
        // Timer simulation for real-time streaming audio callback
        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] timer in
            guard let self = self, self.isListening else {
                timer.invalidate()
                return
            }
            self.currentAudioLevel = Float.random(in: 0.2...0.85)
            let dummyPCM = (0..<8000).map { _ in Float.random(in: -0.5...0.5) }
            let spec = self.preprocessor.convertTo128MelSpectrogram(pcmBuffer: dummyPCM)
            self.onAudioChunkProcessed?(spec)
        }
    }
    
    public func stopListening() {
        isListening = false
        currentAudioLevel = 0.0
    }
}
