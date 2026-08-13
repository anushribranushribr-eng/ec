import SwiftUI

public struct SettingsView: View {
    @AppStorage("confidenceThreshold") private var confidenceThreshold: Double = 0.45
    @AppStorage("enableVAD") private var enableVAD: Bool = true
    @AppStorage("smoothingAlpha") private var smoothingAlpha: Double = 0.35
    
    public var body: some View {
        Form {
            Section(header: Text("Model Parameters")) {
                VStack(alignment: .leading) {
                    Text("Confidence Threshold: \(Int(confidenceThreshold * 100))%")
                    Slider(value: $confidenceThreshold, in: 0.30...0.80, step: 0.05)
                }
                
                VStack(alignment: .leading) {
                    Text("Temporal Smoothing Alpha: \(String(format: "%.2f", smoothingAlpha))")
                    Slider(value: $smoothingAlpha, in: 0.10...0.80, step: 0.05)
                }
                
                Toggle("Voice/Cry Activity Detection (VAD)", isOn: $enableVAD)
            }
            
            Section(header: Text("Safety Disclaimer")) {
                Text("DeepInfant provides an experimental AI-based interpretation of infant vocalizations. It is not a medical device and cannot diagnose illness, pain, or medical conditions.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .navigationTitle("Settings")
    }
}
