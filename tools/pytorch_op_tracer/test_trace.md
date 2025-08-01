# UniAD PyTorch Operation Trace Report

**Model**: None
**Stage**: 2
**Trace Time**: 0.08 seconds

## Summary

- Total Operations: 7
- Unique Operations: 5
- Total Memory: 73.5 MB
- Total Compute Time: 4.7 ms

## Task Head Analysis


## Dataflow Visualization

```mermaid
graph TB
    subgraph "Temporal Queue"
        Input[Multi-Frame Input] --> BEV[BEV Features]
    end

    subgraph "Task Heads"
    end

    BEV --> Track
    BEV --> Seg
    Track --> Motion
    Track --> Occ
    Motion --> Planning
    Occ --> Planning

    style Track fill:#f9f,stroke:#333,stroke-width:2px
    style Motion fill:#bbf,stroke:#333,stroke-width:2px
    style Planning fill:#bfb,stroke:#333,stroke-width:2px
```

