import SwiftUI

public struct AnalysisView: View {
    @StateObject private var viewModel = AnalysisViewModel()
    
    public var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Warning Banner
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text("Assistive AI Tool Notice")
                            .font(.subheadline)
                            .bold()
                    }
                    Text("DeepInfant provides an experimental AI-based interpretation of infant vocalizations. It is not a medical device and cannot diagnose illness, pain, or hunger. Always seek professional care if concerned.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(RoundedRectangle(cornerRadius: 12).fill(Color.orange.opacity(0.1)))
                
                // Live Listening Card
                VStack(spacing: 16) {
                    Text(viewModel.isListening ? "Listening to Audio..." : "Tap to Start Listening")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Button(action: {
                        viewModel.toggleListening()
                    }) {
                        ZStack {
                            Circle()
                                .fill(viewModel.isListening ? Color.red : Color.blue)
                                .frame(width: 80, height: 80)
                            
                            Image(systemName: viewModel.isListening ? "stop.fill" : "mic.fill")
                                .font(.title)
                                .foregroundColor(.white)
                        }
                    }
                    
                    if viewModel.isListening {
                        ProgressView(value: Double(viewModel.audioLevel))
                            .progressViewStyle(LinearProgressViewStyle(tint: .blue))
                            .padding(.horizontal)
                    }
                }
                .padding()
                .frame(maxWidth: .infinity)
                .background(RoundedRectangle(cornerRadius: 16).fill(Color(.tertiarySystemBackground)))
                
                // Results Card
                if let result = viewModel.currentResult {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Possible reason")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Text(result.primaryClass.displayName)
                            .font(.largeTitle)
                            .bold()
                            .foregroundColor(.primary)
                        
                        HStack {
                            Text("Confidence:")
                            Text("\(Int(result.confidence * 100))%")
                                .bold()
                        }
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        
                        Text(result.primaryClass.neutralDescription)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .padding(.top, 4)
                        
                        ProbabilityBarsView(probabilities: result.probabilities)
                            .padding(.top, 10)
                    }
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 16).fill(Color(.secondarySystemBackground)))
                }
            }
            .padding()
        }
        .navigationTitle("DeepInfant Live Analysis")
    }
}
