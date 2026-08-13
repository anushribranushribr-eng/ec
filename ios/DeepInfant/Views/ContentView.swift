import SwiftUI

public struct ContentView: View {
    public var body: some View {
        TabView {
            NavigationView {
                AnalysisView()
            }
            .tabItem {
                Label("Live Analysis", systemImage: "waveform")
            }
            
            NavigationView {
                SettingsView()
            }
            .tabItem {
                Label("Settings", systemImage: "gearshape")
            }
        }
    }
}
