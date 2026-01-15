# Steel Mill Process Overview (steel_mill.png)

This document provides a detailed description of the `images/steel_mill.png` diagram for LLM context.

## Overview

The image shows the **complete integrated steel production process** from raw materials to finished products. It illustrates the broader industrial context in which the torpedo scheduling problem exists.

## The Four Main Stages

Steel production flows through four major stages, left to right:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│              │    │              │    │              │    │              │
│  IRON        │───▶│  STEEL       │───▶│  CONTINUOUS  │───▶│  HOT STRIP   │
│  MAKING      │    │  MAKING      │    │  CASTING     │    │  MILL        │
│              │    │              │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                          ▲
                          │
                    OUR PROBLEM
                    DOMAIN HERE
```

---

## Stage 1: Iron Making (Blast Furnace)

**Location in image:** Leftmost section

**Process:**
- Raw materials: Iron ore, coke (carbon), limestone
- Heated to ~1500°C in a tall blast furnace
- Chemical reduction: Iron oxide → Molten iron
- Output: Hot metal (liquid iron) with impurities

**Output characteristics:**
- Temperature: ~1400-1500°C
- Contains: Carbon (~4%), Sulfur, Silicon, Manganese
- Sulfur level: Varies (rated 1-5 in our problem, where 5 = high sulfur)

**Scheduling relevance:**
- `metalStartsPouringAt`: Times when hot metal is ready
- `metalSulfurLevels`: Quality of each batch (1-5 scale)
- Production is **continuous** - metal MUST be removed on schedule

---

## Stage 2: Steel Making (Our Problem Domain)

**Location in image:** Center-left section

**This is where TORPEDO SCHEDULING happens!**

### Components shown:

#### Torpedo Cars
- Cigar-shaped rail vehicles (~300 tonnes capacity)
- Transport molten iron while keeping it hot
- Limited fleet must be reused efficiently

#### Desulfurization Station
- Chemical treatment to reduce sulfur content
- Injects reagents (magnesium, calcium carbide)
- Takes time: `timeToDesulfOneLevel` per sulfur level reduction
- Limited capacity: `numOfSlotsAtDesulf` simultaneous treatments

#### Oxygen Converter (BOF - Basic Oxygen Furnace)
- Blows pure oxygen into molten iron
- Removes carbon (4% → 0.1-1%)
- Removes other impurities
- Processing time: `timeToConvert`
- Scheduled events: `converterOpensAt` (fixed times)
- Quality requirement: `converterMaxSulfurLevels`

### The Scheduling Challenge:
```
Blast Furnace Events          Converter Events
(production schedule)         (demand schedule)
        │                            │
        ▼                            ▼
   [5, 15, 25, 47, 70]         [30, 57, 62, 80]
        │                            │
        └──────── MATCH ─────────────┘
                   │
          Assign torpedoes
          Route through network
          Meet timing constraints
          Minimize: missed conversions, torpedoes, desulf time
```

---

## Stage 3: Continuous Casting

**Location in image:** Center-right section

**Process:**
- Liquid steel poured into mold
- Cooled by water sprays
- Solidifies into continuous strand
- Cut into semi-finished products

**Output forms:**
- **Slabs**: Flat, rectangular (for sheets/plates)
- **Blooms**: Square/rectangular (for beams/rails)
- **Billets**: Small square (for bars/wire)

**Scheduling relevance:**
- Downstream process - not part of our problem
- But converter timing affects this stage
- Missed conversions → production delays downstream

---

## Stage 4: Hot Strip Mill

**Location in image:** Rightmost section

**Process:**
- Reheats semi-finished steel
- Passes through series of rollers
- Progressively thins and shapes the material

**Output products:**
- Hot rolled coils
- Steel sheets
- Plates
- Various profiles

**Scheduling relevance:**
- Final stage - not part of our problem
- Quality of upstream scheduling affects throughput

---

## Visual Elements in the Diagram

### Equipment Representations:
| Symbol | Meaning |
|--------|---------|
| Large vessel with flames | Blast Furnace |
| Cylindrical horizontal vessel | Torpedo car |
| Vessel with oxygen lance | Converter (BOF) |
| Vertical mold structure | Continuous caster |
| Series of rollers | Rolling mill |

### Flow Indicators:
- **Arrows**: Material flow direction
- **Railway tracks**: Torpedo car routes
- **Pipes/channels**: Molten metal transfer

### Temperature Indicators:
- **Red/orange glow**: Molten material (~1400°C+)
- **Cooling water sprays**: At continuous caster

---

## Why This Context Matters

Understanding the full process helps explain:

1. **Why timing is critical**: 
   - Blast furnace runs continuously (can't stop)
   - Converters have scheduled batches
   - Metal cools if delayed too long

2. **Why minimizing torpedoes matters**:
   - Torpedo cars are expensive capital equipment
   - Maintenance and operational costs
   - Fewer torpedoes = lower investment

3. **Why desulfurization time matters**:
   - Chemical reagents are expensive
   - Station capacity is limited
   - Faster throughput = higher production

4. **Why emergency pit exists**:
   - Production imbalances happen
   - Better to dump excess than shut down blast furnace
   - Material can be recycled (not completely wasted)

---

## Scale Reference

Typical real-world dimensions:
- Blast furnace height: 30-40 meters
- Torpedo car length: ~15-20 meters
- Torpedo car capacity: 200-350 tonnes of molten iron
- Converter capacity: 100-400 tonnes per heat
- Temperature of molten iron: 1400-1500°C
- Complete torpedo trip: 1-3 hours

---

## Connection to Problem Input

The diagram helps interpret the JSON input parameters:

```json
{
  // STAGE 2 ONLY - This is what we're scheduling
  
  // Blast Furnace outputs (left side of our domain)
  "metalStartsPouringAt": [...],    // When iron is ready
  "metalSulfurLevels": [...],       // Quality of each batch
  
  // Converter demands (right side of our domain)  
  "converterOpensAt": [...],        // When converter expects metal
  "converterMaxSulfurLevels": [...], // Required quality
  
  // Network parameters (middle infrastructure)
  "numOfSlotsAtFullBuffer": N,
  "numOfSlotsAtDesulf": M,
  "numOfSlotsAtConverter": K,
  "timeToTransfer...": ...          // Railway segment times
}
```

