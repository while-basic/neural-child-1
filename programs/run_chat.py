from child_model import DynamicNeuralChild
from main import MotherLLM
from chat_interface import NeuralChildInterface

def main():
    print("🚀 Initializing Neural Child Development System...")
    
    # Initialize core components
    child = DynamicNeuralChild()
    mother = MotherLLM()
    
    # Create and launch interface
    interface = NeuralChildInterface(child, mother)
    
    # Launch the interface
    print("\n💬 Launching chat interface...")
    interface_server = interface.create_interface().launch(
        server_name="localhost",
        server_port=7860,
        share=False  # Set to True if you want a public URL
    )
    
    print("\n✨ Neural Child Development System is ready!")
    print("Access the interface at: http://localhost:7860")
    print("Press Ctrl+C to shut down the system.")
    
    try:
        # Keep the server running
        interface_server.block_thread()
    except KeyboardInterrupt:
        print("\nShutting down Neural Child Development System...")
        interface_server.close()
        print("System shutdown complete.")

if __name__ == "__main__":
    main() 