"use client"

import { useState } from "react"
import { Button } from "./button"
import { Input } from "./input"
import { Upload, CheckCircle, AlertCircle } from "lucide-react"
import { toast } from "sonner"

interface ModelUploadProps {
  className?: string
}

export function ModelUpload({ className }: ModelUploadProps): JSX.Element {
  const [isUploading, setIsUploading] = useState<boolean>(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [modelName, setModelName] = useState<string>("")

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      if (!file.name.match(/\.(pt|pth|ckpt)$/)) {
        toast.error("Invalid file type. Only .pt, .pth, or .ckpt files are allowed.")
        return
      }
      setSelectedFile(file)
      setModelName(file.name)
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) {
      toast.error("Please select a model file first")
      return
    }

    setIsUploading(true)
    const formData = new FormData()
    formData.append("model_file", selectedFile)
    if (modelName) {
      formData.append("model_name", modelName)
    }

    try {
      const response = await fetch("/api/models/upload", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`)
      }

      const data = await response.json()
      toast.success("Model uploaded successfully")
      setSelectedFile(null)
      setModelName("")
    } catch (error) {
      toast.error(`Failed to upload model: ${error instanceof Error ? error.message : "Unknown error"}`)
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className={`flex flex-col gap-4 p-4 border rounded-lg ${className}`}>
      <div className="flex flex-col gap-2">
        <label htmlFor="model-file" className="text-sm font-medium">
          Select Model File
        </label>
        <Input
          id="model-file"
          type="file"
          accept=".pt,.pth,.ckpt"
          onChange={handleFileSelect}
          className="cursor-pointer"
        />
      </div>

      <div className="flex flex-col gap-2">
        <label htmlFor="model-name" className="text-sm font-medium">
          Model Name (optional)
        </label>
        <Input
          id="model-name"
          type="text"
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          placeholder="Custom name for the model"
        />
      </div>

      <Button
        onClick={handleUpload}
        disabled={!selectedFile || isUploading}
        className="w-full"
      >
        {isUploading ? (
          <>
            <Upload className="mr-2 h-4 w-4 animate-spin" />
            Uploading...
          </>
        ) : (
          <>
            <Upload className="mr-2 h-4 w-4" />
            Upload Model
          </>
        )}
      </Button>
    </div>
  )
} 