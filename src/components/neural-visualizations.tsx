"use client"

import { useEffect, useState } from "react"
import dynamic from "next/dynamic"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

// Dynamic import for Plotly
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false })

interface NeuralActivityData {
  timestamp: number
  activity_values: number[]
  mean_activation: number
  spike_rate: number
  network_load: number
}

interface NetworkTopologyData {
  node_positions: number[][]
  edge_connections: number[][]
  node_activations: number[]
  edge_weights: number[]
}

interface NeuralMetrics {
  activity: NeuralActivityData
  topology: NetworkTopologyData
}

export function NeuralVisualizations() {
  const [activityHistory, setActivityHistory] = useState<{
    timestamps: number[]
    values: number[]
  }>({
    timestamps: [],
    values: []
  })
  
  const [topologyData, setTopologyData] = useState<NetworkTopologyData | null>(null)
  const MAX_HISTORY_POINTS = 100

  useEffect(() => {
    const eventSource = new EventSource("/api/neural/stream")

    eventSource.onmessage = (event) => {
      const data: NeuralMetrics = JSON.parse(event.data)
      
      // Update activity history
      setActivityHistory(prev => {
        const newTimestamps = [...prev.timestamps, data.activity.timestamp]
        const newValues = [...prev.values, data.activity.mean_activation]
        
        // Keep only the last MAX_HISTORY_POINTS
        if (newTimestamps.length > MAX_HISTORY_POINTS) {
          newTimestamps.shift()
          newValues.shift()
        }
        
        return {
          timestamps: newTimestamps,
          values: newValues
        }
      })
      
      // Update topology
      setTopologyData(data.topology)
    }

    eventSource.onerror = (error) => {
      console.error("EventSource failed:", error)
      eventSource.close()
    }

    return () => {
      eventSource.close()
    }
  }, [])

  const activityTrace = {
    x: activityHistory.timestamps.map(t => new Date(t * 1000)),
    y: activityHistory.values,
    type: "scatter",
    mode: "lines",
    name: "Neural Activity",
    line: { color: "white", width: 2 }
  }

  const topologyTraces = topologyData ? [
    // Nodes
    {
      type: "scatter3d",
      mode: "markers",
      x: topologyData.node_positions.map(pos => pos[0]),
      y: topologyData.node_positions.map(pos => pos[1]),
      z: topologyData.node_positions.map(pos => pos[2]),
      marker: {
        size: 6,
        color: topologyData.node_activations,
        colorscale: "Viridis",
        opacity: 0.8
      },
      name: "Nodes"
    },
    // Edges
    {
      type: "scatter3d",
      mode: "lines",
      x: topologyData.edge_connections.flatMap(([i, j]) => [
        topologyData.node_positions[i][0],
        topologyData.node_positions[j][0],
        null
      ]),
      y: topologyData.edge_connections.flatMap(([i, j]) => [
        topologyData.node_positions[i][1],
        topologyData.node_positions[j][1],
        null
      ]),
      z: topologyData.edge_connections.flatMap(([i, j]) => [
        topologyData.node_positions[i][2],
        topologyData.node_positions[j][2],
        null
      ]),
      line: {
        color: "rgba(255, 255, 255, 0.2)",
        width: 1
      },
      name: "Connections"
    }
  ] : []

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Card className="bg-black border border-white/10">
        <CardHeader>
          <CardTitle>Neural Network Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <Plot
            data={[activityTrace]}
            layout={{
              paper_bgcolor: "black",
              plot_bgcolor: "black",
              font: { color: "white" },
              xaxis: {
                gridcolor: "rgba(255,255,255,0.1)",
                zerolinecolor: "rgba(255,255,255,0.1)",
                title: "Time"
              },
              yaxis: {
                gridcolor: "rgba(255,255,255,0.1)",
                zerolinecolor: "rgba(255,255,255,0.1)",
                title: "Activity",
                range: [0, 1]
              },
              margin: { t: 20, r: 20, b: 40, l: 40 },
              showlegend: false,
              autosize: true
            }}
            config={{ responsive: true }}
            className="w-full h-[300px]"
          />
        </CardContent>
      </Card>

      <Card className="bg-black border border-white/10">
        <CardHeader>
          <CardTitle>Network Topology</CardTitle>
        </CardHeader>
        <CardContent>
          <Plot
            data={topologyTraces}
            layout={{
              paper_bgcolor: "black",
              plot_bgcolor: "black",
              scene: {
                xaxis: { showgrid: false, zeroline: false, showticklabels: false },
                yaxis: { showgrid: false, zeroline: false, showticklabels: false },
                zaxis: { showgrid: false, zeroline: false, showticklabels: false },
                bgcolor: "black",
                camera: {
                  eye: { x: 1.5, y: 1.5, z: 1.5 }
                }
              },
              margin: { t: 0, r: 0, b: 0, l: 0 },
              showlegend: false,
              autosize: true
            }}
            config={{ responsive: true }}
            className="w-full h-[400px]"
          />
        </CardContent>
      </Card>
    </div>
  )
} 