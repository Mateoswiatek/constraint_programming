# Torpedo Routing Network Diagram (overview.png)

This document provides a detailed description of the `images/overview.png` diagram for LLM context.

## Visual Layout

The diagram shows a **directed graph** representing the torpedo car routing network in a steel mill facility. The layout flows generally left-to-right with a return loop.

```
                              ┌─────────────────┐
                              │  EMERGENCY PIT  │
                              │   (unlimited)   │
                              └────────▲────────┘
                                       │
                                       │ emergency route
                                       │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│             │    │             │    │             │    │             │    │             │
│   EMPTY     │───▶│   BLAST     │───▶│    FULL     │───▶│  DESULFUR-  │───▶│  CONVERTER  │
│   BUFFER    │    │   FURNACE   │    │   BUFFER    │    │   IZATION   │    │             │
│             │    │             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       ▲                                                                           │
       │                                                                           │
       └───────────────────────────────────────────────────────────────────────────┘
                                    return path
```

## Nodes (Facilities)

Each node represents a physical location where torpedoes can be present:

| Node | Description | Capacity | Activity |
|------|-------------|----------|----------|
| **Empty Buffer** | Parking area for empty torpedo cars waiting for assignment | **Unlimited (∞)** | Torpedoes wait here between trips |
| **Blast Furnace** | Where molten iron is poured into torpedo | **1 torpedo** | Loading takes `timeToPourMetal` time units |
| **Full Buffer** | Intermediate parking for full torpedoes | **`numOfSlotsAtFullBuffer`** | Optional waiting, can pass through instantly |
| **Desulfurization** | Chemical treatment station to reduce sulfur level | **`numOfSlotsAtDesulf`** | Reducing sulfur by 1 level takes `timeToDesulfOneLevel` |
| **Converter** | Oxygen converter where iron becomes steel | **`numOfSlotsAtConverter`** | Processing takes `timeToConvert` time units |
| **Emergency Pit** | Dump location for excess molten iron | **Unlimited (∞)** | Used when converter capacity exceeded |

## Edges (Railway Segments)

Each edge represents a railway track segment. **Critical constraint: Only ONE torpedo can occupy any segment at a time.**

| Segment | From → To | Travel Time Parameter | Capacity |
|---------|-----------|----------------------|----------|
| 1 | Empty Buffer → Blast Furnace | `timeToTransferFromEmptyBufferToFurnace` | 1 |
| 2 | Blast Furnace → Full Buffer | `timeToTransferFromFurnaceToFullBuffer` | 1 |
| 3 | Full Buffer → Desulfurization | `timeToTransferFromFullBufferToDesulf` | 1 |
| 4 | Desulfurization → Converter | `timeToTransferFromDesulfToConverter` | 1 |
| 5 | Converter → Empty Buffer | `timeToTransferFromConverterToEmptyBuffer` | 1 |
| 6 | Blast Furnace → Empty Buffer | `timeToEmergencyTransferFromFurnaceToEmptyBuffer` | 1 |

## Two Possible Routes

### Route 1: Normal Delivery (Converter Trip)
```
Empty Buffer → Blast Furnace → Full Buffer → Desulfurization → Converter → Empty Buffer
```
- Picks up molten iron at blast furnace
- Optionally waits at full buffer
- Optionally reduces sulfur level at desulfurization station
- Delivers to converter at scheduled time
- Returns empty to buffer

### Route 2: Emergency Pit Trip
```
Empty Buffer → Blast Furnace → Emergency Pit → Empty Buffer
```
- Picks up molten iron at blast furnace
- Dumps iron at emergency pit (no delivery to converter)
- Returns empty to buffer
- Used when: more blast furnace events than converter events, or timing infeasible

## Timing Constraints Illustrated

For a torpedo to deliver metal from blast furnace event `i` to converter event `j`:

```
Timeline:
────────────────────────────────────────────────────────────────────▶ time

    start_trip          metal_starts[i]                    conv_opens[j]
        │                     │                                  │
        ▼                     ▼                                  ▼
   ┌─────────┐ ┌─────────────────┐ ┌─────┐ ┌─────────┐ ┌────────────────┐ ┌─────────┐
   │ travel  │ │ wait + pour     │ │travel│ │ desulf  │ │ wait + convert │ │ travel  │
   │ to BF   │ │ at furnace      │ │ → FB │ │         │ │ at converter   │ │ to EB   │
   └─────────┘ └─────────────────┘ └─────┘ └─────────┘ └────────────────┘ └─────────┘
                                                              │
                                                              │
                                              Must arrive BY conv_opens[j]
```

**Key constraint:** The torpedo must arrive at the converter **before or at** `conv_opens[j]`. It then waits until the converter opens, processes for `timeToConvert`, and returns.

## Capacity Visualization

At any point in time, the following limits must be respected:

```
                    Emergency Pit: ∞ (no limit)
                           │
Empty Buffer: ∞ ──── Blast Furnace: 1 ──── Full Buffer: N ──── Desulf: M ──── Converter: K
      │                    │                    │                  │               │
      │                    │                    │                  │               │
      └────────────────────┴────────────────────┴──────────────────┴───────────────┘
                           All railway segments: capacity 1 (except emergency)
```

## Color Coding (if present in image)

Typical coloring convention:
- **Blue/Gray boxes**: Facilities/stations
- **Arrows/Lines**: Railway segments
- **Dashed line**: Emergency route (alternative path)
- **Numbers on arrows**: Travel times or capacities

## Relevance to Optimization

The diagram helps understand:
1. **Sequencing constraints**: Torpedoes must visit nodes in order
2. **Capacity bottlenecks**: Furnace (1 slot) and railway segments (1 torpedo) are tight
3. **Flexibility points**: Full buffer allows waiting/reordering
4. **Trade-off**: Emergency route is faster but wastes material (missed conversion)

