from langsmith import Client
from sentinel_system import app # Imports your graph
from langchain_core.messages import HumanMessage

client = Client()
dataset_name = "PSCA_Emergency_Scenarios"

# 1. Create a Dataset of common PSCA scenarios
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        inputs=[
            {"input": "Reckless driving reported on Mall Road, Black Sedan."},
            {"input": "Suspicious activity near Metro Station, White Van."},
            {"input": "Accident at Kalma Chowk, two motorcycles involved."}
        ],
        dataset_id=dataset.id
    )

# 2. Run the Graph against the Dataset
print("Starting LangSmith Benchmark...")
for example in client.list_examples(dataset_name=dataset_name):
    config = {"configurable": {"thread_id": f"test_{example.id}"}}
    app.invoke({
        "messages": [HumanMessage(content=example.inputs["input"])],
        "vision_checked": False,
        "intel_checked": False,
        "compliance_approved": False
    }, config=config)

print(f"Benchmark complete. View results at: https://smith.langchain.com/o/default/projects/p/PSCA-Sentinel-Hybrid")