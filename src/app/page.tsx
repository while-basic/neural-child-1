"use client"

import { useEffect, useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { ModelUpload } from "@/components/ui/model-upload"
import { NeuralVisualizations } from "@/components/neural-visualizations"
import dynamic from "next/dynamic"
import { 
  Brain, Activity, AlertTriangle, Loader2, 
  Gauge, Waves, Cpu, AlertOctagon, 
  ShieldAlert, Siren, Pause, Play, Lock,
  BookOpen, Sparkles, Heart, Zap,
  Settings, BarChart, Network, Baby,
  Wifi, WifiOff
} from "lucide-react"
import { useDevelopmentStore } from "@/lib/store"
import { toast } from "sonner"

// Dynamic import for Plotly
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false })

interface DevelopmentStage {
  name: string
  progress: number
  milestones: Array<{
    name: string
    completed: boolean
  }>
}

interface SystemState {
  paused: boolean
  emergencyStop: boolean
  safetyOverrides: {
    disableWarnings: boolean
    allowOverstimulation: boolean
    bypassSafetyLimits: boolean
  }
}

interface ConnectionState {
  status: 'connecting' | 'connected' | 'disconnected'
  lastConnected: Date | null
  retryCount: number
}

export default function Dashboard() {
  const {
    emotionalState,
    warnings,
    developmentSpeed,
    currentStage,
    ageMonths,
    isLoading,
    error,
    startRealTimeUpdates,
    stopRealTimeUpdates,
    fetchInitialState,
  } = useDevelopmentStore()

  const [systemState, setSystemState] = useState<SystemState>({
    paused: false,
    emergencyStop: false,
    safetyOverrides: {
      disableWarnings: false,
      allowOverstimulation: false,
      bypassSafetyLimits: false
    }
  })

  const [connectionState, setConnectionState] = useState<ConnectionState>({
    status: 'connecting',
    lastConnected: null,
    retryCount: 0
  })

  useEffect(() => {
    const initializeConnection = async () => {
      try {
        setConnectionState(prev => ({ ...prev, status: 'connecting' }))
        await fetchInitialState()
        startRealTimeUpdates()
        setConnectionState({
          status: 'connected',
          lastConnected: new Date(),
          retryCount: 0
        })
        toast.success("Connected to Neural Development System", {
          description: "Real-time monitoring active"
        })
      } catch (err) {
        setConnectionState(prev => ({
          status: 'disconnected',
          lastConnected: prev.lastConnected,
          retryCount: prev.retryCount + 1
        }))
        toast.error("Connection Failed", {
          description: "Attempting to reconnect..."
        })
        // Retry connection after 5 seconds
        setTimeout(initializeConnection, 5000)
      }
    }

    initializeConnection()
    return () => stopRealTimeUpdates()
  }, [])

  useEffect(() => {
    if (error) {
      setConnectionState(prev => ({
        status: 'disconnected',
        lastConnected: prev.lastConnected,
        retryCount: prev.retryCount + 1
      }))
      toast.error(error)
    }
  }, [error])

  if (isLoading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-black">
        <div className="flex flex-col items-center space-y-4">
          <Loader2 className="h-8 w-8 animate-spin text-white" />
          <p className="text-white">Initializing Neural Development System...</p>
          <p className="text-white/60 text-sm">Establishing connection...</p>
        </div>
      </div>
    )
  }

  const handleEmergencyStop = () => {
    setSystemState(prev => ({
      ...prev,
      paused: true,
      emergencyStop: true
    }))
    stopRealTimeUpdates()
    toast.error("EMERGENCY STOP ACTIVATED", {
      description: "Neural development processes halted."
    })
  }

  const handleSystemResume = () => {
    setSystemState(prev => ({
      ...prev,
      paused: false,
      emergencyStop: false
    }))
    startRealTimeUpdates()
    toast.success("System Resumed")
  }

  return (
    <div className="min-h-screen bg-black text-white p-6">
      {/* Header with Emergency Controls and Connection Status */}
      <div className="flex justify-between items-center mb-8">
        <div className="flex items-center space-x-4">
          <Baby className="h-8 w-8" />
          <h1 className="text-2xl font-bold">Neural Development System</h1>
          <div className={`px-3 py-1 rounded text-sm flex items-center space-x-2 ${
            connectionState.status === 'connected' 
              ? 'bg-white/10 text-green-500' 
              : connectionState.status === 'connecting'
              ? 'bg-white/10 text-yellow-500'
              : 'bg-white/10 text-red-500'
          }`}>
            {connectionState.status === 'connected' ? (
              <>
                <Wifi className="h-4 w-4" />
                <span>Connected</span>
              </>
            ) : connectionState.status === 'connecting' ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Connecting{connectionState.retryCount > 0 ? ` (Attempt ${connectionState.retryCount})` : ''}</span>
              </>
            ) : (
              <>
                <WifiOff className="h-4 w-4" />
                <span>Disconnected</span>
              </>
            )}
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <div className={`px-4 py-2 rounded border ${
            warnings.warning_state === "GREEN" ? "border-green-500 text-green-500" :
            warnings.warning_state === "YELLOW" ? "border-yellow-500 text-yellow-500" :
            "border-red-500 text-red-500"
          }`}>
            <AlertTriangle className="inline-block mr-2 h-5 w-5" />
            {warnings.warning_state} Status
          </div>
          <button
            onClick={systemState.paused ? handleSystemResume : handleEmergencyStop}
            className={`px-6 py-2 rounded font-bold flex items-center ${
              systemState.paused
                ? "bg-green-500 hover:bg-green-600"
                : "bg-red-500 hover:bg-red-600"
            }`}
            disabled={connectionState.status !== 'connected'}
          >
            {systemState.paused ? (
              <><Play className="mr-2 h-5 w-5" /> Resume</>
            ) : (
              <><Pause className="mr-2 h-5 w-5" /> Emergency Stop</>
            )}
          </button>
        </div>
      </div>

      {/* Connection Lost Warning */}
      {connectionState.status === 'disconnected' && (
        <div className="mb-8 p-4 border border-red-500 rounded bg-black text-red-500 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <AlertOctagon className="h-5 w-5" />
            <span>Connection lost to Neural Development System</span>
          </div>
          <button
            onClick={() => {
              setConnectionState(prev => ({ ...prev, status: 'connecting' }))
              fetchInitialState()
            }}
            className="px-4 py-2 rounded bg-red-500 text-white hover:bg-red-600"
          >
            Retry Connection
          </button>
        </div>
      )}

      {/* Core Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <Card className="bg-black border border-white/10">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-sm">Development Stage</CardTitle>
            <Brain className="h-4 w-4 text-white/60" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{currentStage}</div>
            <Progress value={75} className="mt-2" />
          </CardContent>
        </Card>

        <Card className="bg-black border border-white/10">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-sm">Neural Age</CardTitle>
            <Activity className="h-4 w-4 text-white/60" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{ageMonths.toFixed(1)} months</div>
            <div className="text-xs text-white/60 mt-1">Accelerated Development</div>
          </CardContent>
        </Card>

        <Card className="bg-black border border-white/10">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-sm">System Stability</CardTitle>
            <Gauge className="h-4 w-4 text-white/60" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(warnings.metrics.emotional_stability * 100).toFixed(1)}%
            </div>
            <Progress 
              value={warnings.metrics.emotional_stability * 100} 
              className="mt-2"
            />
          </CardContent>
        </Card>

        <Card className="bg-black border border-white/10">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-sm">Development Speed</CardTitle>
            <Zap className="h-4 w-4 text-white/60" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{developmentSpeed.toFixed(1)}x</div>
            <div className="text-xs text-white/60 mt-1">Acceleration Factor</div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid grid-cols-2 lg:grid-cols-6 gap-2 bg-black border border-white/10 p-1">
          <TabsTrigger value="overview" className="data-[state=active]:bg-white data-[state=active]:text-black">
            <BarChart className="h-4 w-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="neural" className="data-[state=active]:bg-white data-[state=active]:text-black">
            <Network className="h-4 w-4 mr-2" />
            Neural
          </TabsTrigger>
          <TabsTrigger value="emotional" className="data-[state=active]:bg-white data-[state=active]:text-black">
            <Heart className="h-4 w-4 mr-2" />
            Emotional
          </TabsTrigger>
          <TabsTrigger value="cognitive" className="data-[state=active]:bg-white data-[state=active]:text-black">
            <Brain className="h-4 w-4 mr-2" />
            Cognitive
          </TabsTrigger>
          <TabsTrigger value="development" className="data-[state=active]:bg-white data-[state=active]:text-black">
            <BookOpen className="h-4 w-4 mr-2" />
            Development
          </TabsTrigger>
          <TabsTrigger value="settings" className="data-[state=active]:bg-white data-[state=active]:text-black">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </TabsTrigger>
          <TabsTrigger value="system" className="data-[state=active]:bg-white data-[state=active]:text-black">
            <Cpu className="h-4 w-4 mr-2" />
            System
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Neural Network Activity */}
            <Card className="bg-black border border-white/10">
              <CardHeader>
                <CardTitle>Neural Network Activity</CardTitle>
              </CardHeader>
              <CardContent>
                <Plot
                  data={[{
                    type: 'scatter',
                    mode: 'lines',
                    x: Array.from({length: 100}, (_, i) => i),
                    y: Array.from({length: 100}, () => Math.random()),
                    line: { color: 'white' }
                  }]}
                  layout={{
                    paper_bgcolor: 'black',
                    plot_bgcolor: 'black',
                    font: { color: 'white' },
                    xaxis: {
                      gridcolor: 'rgba(255,255,255,0.1)',
                      zerolinecolor: 'rgba(255,255,255,0.1)'
                    },
                    yaxis: {
                      gridcolor: 'rgba(255,255,255,0.1)',
                      zerolinecolor: 'rgba(255,255,255,0.1)'
                    },
                    margin: { t: 20, r: 20, b: 40, l: 40 }
                  }}
                  config={{ responsive: true }}
                  className="w-full h-[300px]"
                />
              </CardContent>
            </Card>

            {/* System Metrics */}
            <Card className="bg-black border border-white/10">
              <CardHeader>
                <CardTitle>System Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {Object.entries(warnings.metrics).map(([key, value]) => (
                  <div key={key} className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">
                        {key.split('_').map(word => 
                          word.charAt(0).toUpperCase() + word.slice(1)
                        ).join(' ')}
                      </span>
                      <span className="text-sm">{(value * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={value * 100} className="h-2" />
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Neural Tab */}
        <TabsContent value="neural">
          <div className="space-y-4">
            {/* Real-time Neural Visualizations */}
            <NeuralVisualizations />

            {/* Existing Neural Metrics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Neural Network Topology */}
              <Card className="bg-black border border-white/10">
                <CardHeader>
                  <CardTitle>Neural Network Topology</CardTitle>
                </CardHeader>
                <CardContent>
                  <Plot
                    data={[{
                      type: "scatter3d",
                      mode: "markers",
                      x: Array.from({length: 50}, () => Math.random() * 2 - 1),
                      y: Array.from({length: 50}, () => Math.random() * 2 - 1),
                      z: Array.from({length: 50}, () => Math.random() * 2 - 1),
                      marker: {
                        size: 3,
                        color: "white",
                        opacity: 0.8
                      }
                    }]}
                    layout={{
                      paper_bgcolor: "black",
                      plot_bgcolor: "black",
                      scene: {
                        xaxis: { gridcolor: "white", zerolinecolor: "white" },
                        yaxis: { gridcolor: "white", zerolinecolor: "white" },
                        zaxis: { gridcolor: "white", zerolinecolor: "white" },
                        bgcolor: "black"
                      },
                      margin: { t: 0, r: 0, b: 0, l: 0 }
                    }}
                    config={{ responsive: true }}
                    className="w-full h-[400px]"
                  />
                </CardContent>
              </Card>

              {/* Synaptic Activity */}
              <Card className="bg-black border border-white/10">
                <CardHeader>
                  <CardTitle>Synaptic Activity</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm mb-2">Connection Strength</div>
                      <div className="text-2xl font-bold">87.5%</div>
                      <Progress value={87.5} className="mt-2" />
                    </div>
                    <div>
                      <div className="text-sm mb-2">Firing Rate</div>
                      <div className="text-2xl font-bold">124 Hz</div>
                      <Progress value={62} className="mt-2" />
                    </div>
                    <div>
                      <div className="text-sm mb-2">Plasticity</div>
                      <div className="text-2xl font-bold">92.1%</div>
                      <Progress value={92.1} className="mt-2" />
                    </div>
                    <div>
                      <div className="text-sm mb-2">Synchronization</div>
                      <div className="text-2xl font-bold">78.3%</div>
                      <Progress value={78.3} className="mt-2" />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Neural Network Topology */}
            <Card className="bg-black border border-white/10">
              <CardHeader>
                <CardTitle>Neural Network Topology</CardTitle>
              </CardHeader>
              <CardContent>
                <Plot
                  data={[{
                    type: 'scatter3d',
                    mode: 'markers',
                    x: Array.from({length: 50}, () => Math.random() * 2 - 1),
                    y: Array.from({length: 50}, () => Math.random() * 2 - 1),
                    z: Array.from({length: 50}, () => Math.random() * 2 - 1),
                    marker: {
                      size: 3,
                      color: 'white',
                      opacity: 0.8
                    }
                  }]}
                  layout={{
                    paper_bgcolor: 'black',
                    plot_bgcolor: 'black',
                    scene: {
                      xaxis: { gridcolor: 'white', zerolinecolor: 'white' },
                      yaxis: { gridcolor: 'white', zerolinecolor: 'white' },
                      zaxis: { gridcolor: 'white', zerolinecolor: 'white' },
                      bgcolor: 'black'
                    },
                    margin: { t: 0, r: 0, b: 0, l: 0 }
                  }}
                  config={{ responsive: true }}
                  className="w-full h-[400px]"
                />
              </CardContent>
            </Card>

            {/* Synaptic Activity */}
            <Card className="bg-black border border-white/10">
              <CardHeader>
                <CardTitle>Synaptic Activity</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm mb-2">Connection Strength</div>
                    <div className="text-2xl font-bold">87.5%</div>
                    <Progress value={87.5} className="mt-2" />
                  </div>
                  <div>
                    <div className="text-sm mb-2">Firing Rate</div>
                    <div className="text-2xl font-bold">124 Hz</div>
                    <Progress value={62} className="mt-2" />
                  </div>
                  <div>
                    <div className="text-sm mb-2">Plasticity</div>
                    <div className="text-2xl font-bold">92.1%</div>
                    <Progress value={92.1} className="mt-2" />
                  </div>
                  <div>
                    <div className="text-sm mb-2">Synchronization</div>
                    <div className="text-2xl font-bold">78.3%</div>
                    <Progress value={78.3} className="mt-2" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Emotional Tab */}
        <TabsContent value="emotional">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Emotional Radar */}
            <Card className="bg-black border border-white/10">
              <CardHeader>
                <CardTitle>Emotional State Radar</CardTitle>
              </CardHeader>
              <CardContent>
                <Plot
                  data={[{
                    type: 'scatterpolar',
                    r: Object.values(emotionalState),
                    theta: Object.keys(emotionalState).map(k => k.charAt(0).toUpperCase() + k.slice(1)),
                    fill: 'toself',
                    fillcolor: 'rgba(255, 255, 255, 0.1)',
                    line: { color: 'white' }
                  }]}
                  layout={{
                    polar: {
                      radialaxis: { range: [0, 1], showline: false, gridcolor: 'rgba(255,255,255,0.1)' },
                      angularaxis: { showline: false, gridcolor: 'rgba(255,255,255,0.1)' },
                      bgcolor: 'black'
                    },
                    paper_bgcolor: 'black',
                    plot_bgcolor: 'black',
                    font: { color: 'white' },
                    margin: { t: 30, r: 30, b: 30, l: 30 },
                    showlegend: false
                  }}
                  config={{ responsive: true }}
                  className="w-full h-[400px]"
                />
              </CardContent>
            </Card>

            {/* Emotional Metrics */}
            <div className="grid grid-cols-1 gap-4">
              {Object.entries(emotionalState).map(([emotion, value]) => (
                <Card key={emotion} className="bg-black border border-white/10">
                  <CardHeader className="py-2">
                    <CardTitle className="text-sm">{emotion.charAt(0).toUpperCase() + emotion.slice(1)}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center justify-between mb-2">
                      <div className="text-2xl font-bold">{(value * 100).toFixed(1)}%</div>
                      <div className={`h-2 w-2 rounded-full ${
                        value > 0.7 ? "bg-white" :
                        value > 0.3 ? "bg-white/60" :
                        "bg-white/20"
                      }`} />
                    </div>
                    <Progress value={value * 100} className="h-2" />
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </TabsContent>

        {/* Cognitive Tab */}
        <TabsContent value="cognitive">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Learning Progress */}
            <Card className="bg-black border border-white/10">
              <CardHeader>
                <CardTitle>Learning Progress</CardTitle>
              </CardHeader>
              <CardContent>
                <Plot
                  data={[{
                    type: 'scatter',
                    mode: 'lines',
                    x: Array.from({length: 100}, (_, i) => i),
                    y: Array.from({length: 100}, (_, i) => Math.pow(i/100, 2)),
                    line: { color: 'white' }
                  }]}
                  layout={{
                    paper_bgcolor: 'black',
                    plot_bgcolor: 'black',
                    font: { color: 'white' },
                    xaxis: { gridcolor: 'rgba(255,255,255,0.1)', title: 'Time' },
                    yaxis: { gridcolor: 'rgba(255,255,255,0.1)', title: 'Knowledge' },
                    margin: { t: 20, r: 20, b: 40, l: 40 }
                  }}
                  config={{ responsive: true }}
                  className="w-full h-[300px]"
                />
              </CardContent>
            </Card>

            {/* Cognitive Metrics */}
            <Card className="bg-black border border-white/10">
              <CardHeader>
                <CardTitle>Cognitive Development</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <div className="flex justify-between mb-2">
                    <div className="text-sm">Memory Formation</div>
                    <div className="text-sm">94.2%</div>
                  </div>
                  <Progress value={94.2} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <div className="text-sm">Pattern Recognition</div>
                    <div className="text-sm">87.8%</div>
                  </div>
                  <Progress value={87.8} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <div className="text-sm">Problem Solving</div>
                    <div className="text-sm">82.3%</div>
                  </div>
                  <Progress value={82.3} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <div className="text-sm">Abstract Thinking</div>
                    <div className="text-sm">76.5%</div>
                  </div>
                  <Progress value={76.5} className="h-2" />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Development Tab */}
        <TabsContent value="development">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Stage Progress */}
            <Card className="bg-black border border-white/10">
              <CardHeader>
                <CardTitle>Stage Progress</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center">
                  <div>
                    <div className="text-lg font-bold">{currentStage}</div>
                    <div className="text-sm text-white/60">Current Stage</div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold">{ageMonths.toFixed(1)} months</div>
                    <div className="text-sm text-white/60">Neural Age</div>
                  </div>
                </div>
                <Progress value={75} className="h-2 mb-4" />
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm mb-2">Stage Duration</div>
                    <div className="text-2xl font-bold">14.2 days</div>
                  </div>
                  <div>
                    <div className="text-sm mb-2">Next Stage</div>
                    <div className="text-2xl font-bold">~3.8 days</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Milestones */}
            <Card className="bg-black border border-white/10">
              <CardHeader>
                <CardTitle>Development Milestones</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    { name: "Basic Reflexes", completed: true, progress: 100 },
                    { name: "Emotional Recognition", completed: true, progress: 100 },
                    { name: "Pattern Learning", completed: false, progress: 85 },
                    { name: "Memory Formation", completed: false, progress: 72 },
                    { name: "Social Interaction", completed: false, progress: 45 }
                  ].map((milestone) => (
                    <div key={milestone.name} className="space-y-2">
                      <div className="flex justify-between">
                        <div className="flex items-center">
                          <div className={`h-2 w-2 rounded-full mr-2 ${
                            milestone.completed ? "bg-white" : "bg-white/20"
                          }`} />
                          <span className="text-sm">{milestone.name}</span>
                        </div>
                        <span className="text-sm">{milestone.progress}%</span>
                      </div>
                      <Progress value={milestone.progress} className="h-1" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Settings Tab */}
        <TabsContent value="settings">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* System Controls */}
            <Card className="bg-black border border-white/10">
              <CardHeader>
                <CardTitle>System Controls</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {Object.entries(systemState.safetyOverrides).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between">
                    <div>
                      <div className="font-medium">{key.replace(/([A-Z])/g, ' $1').trim()}</div>
                      <div className="text-sm text-white/60">
                        {value ? "Override Active" : "Safety Active"}
                      </div>
                    </div>
                    <button
                      onClick={() => setSystemState(prev => ({
                        ...prev,
                        safetyOverrides: {
                          ...prev.safetyOverrides,
                          [key]: !value
                        }
                      }))}
                      className={`px-4 py-2 rounded text-sm font-medium ${
                        value 
                          ? "bg-red-500 hover:bg-red-600"
                          : "bg-white text-black hover:bg-white/90"
                      }`}
                    >
                      {value ? "Disable Override" : "Enable Override"}
                    </button>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* System Status */}
            <Card className="bg-black border border-white/10">
              <CardHeader>
                <CardTitle>System Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm mb-2">CPU Usage</div>
                    <div className="text-2xl font-bold">32.5%</div>
                    <Progress value={32.5} className="mt-2" />
                  </div>
                  <div>
                    <div className="text-sm mb-2">Memory Usage</div>
                    <div className="text-2xl font-bold">2.1 GB</div>
                    <Progress value={45.8} className="mt-2" />
                  </div>
                  <div>
                    <div className="text-sm mb-2">Network I/O</div>
                    <div className="text-2xl font-bold">1.2 MB/s</div>
                    <Progress value={28.4} className="mt-2" />
                  </div>
                  <div>
                    <div className="text-sm mb-2">Storage</div>
                    <div className="text-2xl font-bold">45.8 GB</div>
                    <Progress value={45.8} className="mt-2" />
                  </div>
                </div>
                <div className="pt-4 border-t border-white/10">
                  <div className="text-sm font-medium mb-4">Active Processes</div>
                  <div className="space-y-2">
                    {[
                      { name: "Neural Processing", status: "Running", load: 28 },
                      { name: "Memory Formation", status: "Running", load: 15 },
                      { name: "Pattern Recognition", status: "Running", load: 22 },
                      { name: "Emotional Processing", status: "Running", load: 18 }
                    ].map((process) => (
                      <div key={process.name} className="flex items-center justify-between">
                        <div>
                          <div className="text-sm">{process.name}</div>
                          <div className="text-xs text-white/60">{process.status}</div>
                        </div>
                        <div className="text-sm">{process.load}%</div>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* System Tab */}
        <TabsContent value="system">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {/* Model Management */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-4 w-4" />
                  Model Management
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ModelUpload />
              </CardContent>
            </Card>
          </div>
        </TabsContent>

      </Tabs>
    </div>
  )
} 