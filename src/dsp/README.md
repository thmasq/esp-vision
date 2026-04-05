# ESP32-S3 PIE (Processor Instruction Extensions) Inline Assembly Manual

## 1. Special Registers: Setup and Execution Context

The ESP32-S3 PIE relies on a hidden state machine. You cannot write efficient inline assembly simply by tracking the 128-bit `QR` registers; you must actively manage the special registers, which dictate shift amounts, accumulation limits, and memory offsets.

### Reading and Writing Special Registers

The PIE unit does not have "load-immediate" or "memory-to-special" instructions for its configuration registers. You must bridge data through the 32-bit General Purpose Address Registers (`AR`, e.g., `a2`, `a3`).

- **`WSR` / `RSR` (Write/Read Special Register):** Used for standard Xtensa registers and the `SAR` (Shift Amount) register.
  - _Example (Setting SAR to 16):_
  ```assembly
  movi a2, 16
  wsr.sar a2 // SAR now tells multipliers to right-shift results by 16 bits
  ```
- **`WUR` / `RUR` (Write/Read User Register):** Used for PIE-specific registers. For registers wider than 32 bits, you must access them in segments using decimal indices or specific suffixes.
  - _Example (Setting SAR_BYTE to 3):_
    ```assembly
    movi a2, 3
    wur.sar_byte a2 // Subsequent EE.SRC.Q calls will shift by 3 bytes (24 bits)
    ```
  - _Example (Accessing the 40-bit ACCX):_
    ```assembly
    rur.accx_0 a2 // Read lower 32 bits of ACCX into a2
    rur.accx_1 a3 // Read upper 8 bits of ACCX into a3
    ```

### The Implicit State Registers

- **`SAR` (Shift Amount Register - 6 bits):**
  - **Usage:** Automatically applied during vector multiplications (`EE.VMUL.*`, `EE.CMUL.*`, `EE.FFT.AMS.*`). It dictates the arithmetic right-shift applied to the 32-bit intermediate multiplication results before they are clamped/truncated to 16-bit or 8-bit outputs.
  - **Execution Dependency:** If you fail to set `SAR`, your vector multiplications will either overflow into meaningless noise (if `SAR` is too low) or zero-out (if `SAR` is too high).
  - **Logic:** `Result = (Input_A * Input_B) >> SAR`.

- **`SAR_BYTE` (Byte Shift Register - 4 bits):**
  - **Usage:** Determines the byte-wise right-shift amount (multiplied internally by 8 to get bits) when splicing unaligned memory blocks using `EE.SRC.Q`.
  - **Execution Dependency:** Can be set manually, but is most commonly populated automatically by the `EE.LD.128.USAR.*` instruction, which captures the alignment offset of a pointer.
  - **Logic:** `Aligned_Vector = (Block_A_and_Block_B_Concatenated) >> (SAR_BYTE * 8)`.

- **`ACCX` (40-bit Global Accumulator):**
  - **Usage:** A single scalar accumulator. Used by `EE.VMULAS.*.ACCX` to sum all parallel lane multiplications into one master value (perfect for dot-products).
  - **Instruction to Clear:** `EE.ZERO.ACCX` (sets the 40-bit value to 0).

- **`QACC_H` and `QACC_L` (160-bit Segmented Accumulators):**
  - **Usage:** Parallel accumulators holding 320 bits of total state. Depending on the instruction width (.8 or .16), they act as sixteen 20-bit accumulators or eight 40-bit accumulators.
  - **Instruction to Clear:** `EE.ZERO.QACC`.
  - **Manual Write:** Use `EE.MOV.S16.QACC qx` to move 128 bits from a `QR` register into the accumulators with sign extension.

- **`FFT_BIT_WIDTH` (4 bits):**
  - **Usage:** Dictates the width of the bit-reversal operation in the `EE.BITREV` instruction.
  - **Mapping:** A value of `0` corresponds to a 3-bit (8-point) reversal; a value of `7` corresponds to a 10-bit (1024-point) reversal.

- **`UA_STATE` (128 bits):**
  - **Usage:** A high-speed memory buffer used exclusively by `EE.FFT.AMS.S16.LD.INCP.UAUP`.
  - **Logic:** It caches the "trailing" 128-bit block from the previous fetch so the current cycle can perform a sliding-window calculation without re-reading the same memory address twice. This is essential for unaligned FFT processing.

---

## 2. The Modifier System (Suffixes)

PIE instructions use a strict suffix hierarchy. Some suffixes dictate data sizes (Mandatory), while others fuse memory operations into the math cycle (Optional).

- **Data Width (Mandatory):** `.8`, `.16`, `.32`, `.64`, `.128`. Dictates the bit-width of the lanes or the total memory block.
- **Signage (Mandatory):** `.U` (Unsigned) or `.S` (Signed).
- **Memory Update (Optional/Mandatory depending on base):**
  - `.IP` (Immediate Post-increment): Adds a hardcoded immediate to the `AR` address register after execution.
  - `.XP` (Index Post-increment): Adds the value of a secondary `AR` register to the address register after execution.
  - `.INCP`: Hardcoded to add `16` to the address register after execution.
  - `.DECP`: Hardcoded to subtract `16` from the address register after execution.
- **Unaligned Processing (Optional):**
  - `.QUP` (Q-register Unaligned Processing): When appended to MAC instructions, tells the hardware to simultaneously execute a `SAR_BYTE` shift concatenation on the fetched data.

---

## 3. Instruction Reference

_Argument Key:_ * `qx, qy, qz` = 128-bit Vector Registers (q0-q7)

- `ar, as, at` = 32-bit General Purpose Registers (a0-a15)
- `imm` = Immediate numerical value
- `sel` = Immediate selector for specific lanes

### 3.1 Memory Read Operations (`LD`)

_All reads force alignment (lower bits to 0) based on fetch size._

- **`EE.VLD.128.[XP | IP]`**
  - _Syntax:_ `EE.VLD.128.IP qx, as, imm` | `EE.VLD.128.XP qx, as, at`
  - _Execution:_ Forces lower 4 bits of `AR` (`as`) to 0. Loads 16 bytes into a 128-bit `QR` register (`qx`). Updates `as` via index (`at`) or immediate (`imm`).
- **`EE.VLD.[H | L].64.[XP | IP]`**
  - _Syntax:_ `EE.VLD.H.64.IP qx, as, imm`
  - _Execution:_ Forces lower 3 bits to 0. Loads 8 bytes. Writes to either the High (`H`) or Low (`L`) 64 bits of a `QR` register.
- **`EE.VLDBC.[8 | 16 | 32]{.XP | .IP}`**
  - _Syntax:_ `EE.VLDBC.8.IP qx, as, imm`
  - _Execution:_ Broadcast load. Fetches 1, 2, or 4 bytes from `as` and duplicates (broadcasts) that value across the entire 128-bit `QR` register `qx`.
- **`EE.LD.128.USAR.[XP | IP]`**
  - _Syntax:_ `EE.LD.128.USAR.IP qx, as, imm`
  - _Execution:_ The crucial unaligned load helper. Fetches 16 bytes into `qx`. **Implicitly overwrites `SAR_BYTE`** with the unaligned lower 4 bits of the original `as` pointer.
- **`EE.LDQA.[U | S][8 | 16].128.[XP | IP]`**
  - _Syntax:_ `EE.LDQA.S16.128.IP as, imm`
  - _Execution:_ Loads 16 bytes. Slices into 8-bit or 16-bit lanes. Sign-extends (`S`) or Zero-extends (`U`) each lane to 20 bits or 40 bits respectively. Drops the massive result directly into the `QACC_L` and `QACC_H` special accumulators.
- **`EE.LDXQ.32`**
  - _Syntax:_ `EE.LDXQ.32 qz, qx, as, imm` (where imm selects lane in qx)
  - _Execution:_ Selects a 16-bit addend from `qx` based on `imm`, shifts it left by 2, adds it to `as`, forces alignment, and fetches 4 bytes into a specified segment of target `qz`. _Modifies address before read._

### 3.2 Memory Write Operations (`ST`)

- **`EE.VST.128.[XP | IP]`**
  - _Syntax:_ `EE.VST.128.IP qx, as, imm`
  - _Execution:_ Forces lower 4 bits of `as` to 0. Stores 16 bytes from `qx` to memory. Updates pointer.
- **`EE.VST.[H | L].64.[XP | IP]`**
  - _Syntax:_ `EE.VST.H.64.IP qx, as, imm`
  - _Execution:_ Stores either the upper or lower 64 bits of `qx`.
- **`EE.ST.ACCX.IP`**
  - _Syntax:_ `EE.ST.ACCX.IP as, imm`
  - _Execution:_ Zero-extends the 40-bit `ACCX` register to 64 bits and stores it at address `as`.
- **`EE.ST.QACC_[H | L].[H.32 | L.128].IP`**
  - _Syntax:_ `EE.ST.QACC_H.L.128.IP as, imm`
  - _Execution:_ Writes data out of the accumulators to memory. Can write a 32-bit upper chunk (`H.32`) or a 128-bit lower chunk (`L.128`).

### 3.3 Data Exchange & Manipulation

- **`EE.MOVI.32.A` / `EE.MOVI.32.Q`**
  - _Syntax:_ `EE.MOVI.32.A ar, qx, sel` | `EE.MOVI.32.Q qx, ar, sel`
  - _Execution:_ Shuttles 32-bit segments between a 128-bit `QR` and a 32-bit `AR`. Requires an immediate selector `sel` (0-3) to target the specific 32-bit lane in the `QR`.
- **`EE.VZIP.[8 | 16 | 32]` / `EE.VUNZIP.[8 | 16 | 32]`**
  - _Syntax:_ `EE.VZIP.16 qx, qy`
  - _Execution:_ Performs interleaving (ZIP) or de-interleaving (UNZIP) of two `QR` registers based on the specified block size. Re-writes the results back into the source registers `qx` and `qy`.
- **`EE.MOV.[U | S][8 | 16].QACC`**
  - _Syntax:_ `EE.MOV.S16.QACC qx`
  - _Execution:_ Takes a standard `QR` register `qx`, slices it, extends the bits (Signed or Zero/Unsigned), and populates `QACC_H` and `QACC_L`.

### 3.4 Vector Arithmetic (SIMD)

- **`EE.VADDS.S[8 | 16 | 32]{.LD.INCP | .ST.INCP}`**
  - _Syntax:_ `EE.VADDS.S16 qz, qx, qy` | `EE.VADDS.S16.LD.INCP qz, qx, qy, qw, as` (qw = memory load target)
  - _Execution:_ Vector addition with **Saturation** (`S`). Adds lanes of `qx` and `qy` into `qz`. If optional memory modifier is used, it simultaneously reads/writes 16 bytes and increments `as` by 16.
- **`EE.VSUBS.S[8 | 16 | 32]{.LD.INCP | .ST.INCP}`**
  - _Syntax:_ `EE.VSUBS.S16 qz, qx, qy`
  - _Execution:_ Vector subtraction with **Saturation**.
- **`EE.VMUL.[U | S][8 | 16]{.LD.INCP | .ST.INCP}`**
  - _Syntax:_ `EE.VMUL.S16 qz, qx, qy`
  - _Execution:_ Multiplies corresponding lanes of `qx` and `qy`. **Implicitly right-shifts the 32-bit intermediate product by `SAR`**. The result wraps around (truncates) to fit the destination lane in `qz`.
- **`EE.CMUL.S16{.LD.INCP | .ST.INCP}`**
  - _Syntax:_ `EE.CMUL.S16 qz, qx, qy, sel4`
  - _Execution:_ Treats 32-bit lanes as `[16-bit Real | 16-bit Imaginary]`. Performs $(Re1 \times Re2 - Im1 \times Im2)$ and $(Re1 \times Im2 + Im1 \times Re2)$. Requires an immediate selector `sel4` (0-3) to target the lane. Implicitly shifts by `SAR`.
- **`EE.VMULAS.[U | S][8 | 16].ACCX{.LD.IP | .LD.XP | .LD.IP.QUP | .LD.XP.QUP}`**
  - _Syntax:_ `EE.VMULAS.S16.ACCX qx, qy` | `EE.VMULAS.S16.ACCX.LD.IP qx, qy, qw, as, imm`
  - _Execution:_ Multiplies lanes of `qx` and `qy`, sums all lane products together, and adds the total to the 40-bit `ACCX` register. The optional `.QUP` variant reads 16 bytes, performs an unaligned shift via `SAR_BYTE`, and uses that new data in the pipeline.
- **`EE.VMULAS.[U | S][8 | 16].QACC{.LD.IP | .LD.XP | .LDBC.INCP | .QUP variants}`**
  - _Syntax:_ `EE.VMULAS.S16.QACC qx, qy`
  - _Execution:_ Multiplies lanes and adds each lane's product to the corresponding segment in `QACC_H`/`QACC_L`. The `.LDBC.INCP` modifier fuses a broadcast load (copies a single byte/halfword across all lanes) into the cycle.
- **`EE.SRCMB.S[8 | 16].QACC`**
  - _Syntax:_ `EE.SRCMB.S16.QACC qz, ar`
  - _Execution:_ The only way to extract accumulator data into a `QR`. Extracts 20-bit/40-bit segments from `QACC`, performs an arithmetic right shift (based on the lower bits of `ar`), **saturates** to 8/16 bits, and writes the cleanly scaled output to `qz`.
- **`EE.SRS.ACCX`**
  - _Syntax:_ `EE.SRS.ACCX ar, as`
  - _Execution:_ Shifts the `ACCX` register right arithmetically by the value in `as`, saturates to 32 bits, and writes to `ar`.

### 3.5 Bitwise, Shift, and Formatting

- **`EE.ORQ`, `EE.XORQ`, `EE.ANDQ`, `EE.NOTQ`**
  - _Syntax:_ `EE.ORQ qz, qx, qy` | `EE.NOTQ qz, qx`
  - _Execution:_ Standard 128-bit wide bitwise logic between `QR` registers.
- **`EE.SRC.Q{.LD.XP | .LD.IP | .QUP}`**
  - _Syntax:_ `EE.SRC.Q qz, qx, qy`
  - _Execution:_ Splicer. Concatenates two 16-byte `QR` registers (`qx`, `qy`) into 32 bytes. Shifts right by `SAR_BYTE * 8` bits. Returns the middle 16-byte aligned window into `qz`. If `.QUP` is used, it saves the upper 8 bytes automatically.
- **`EE.SLCI.2Q` / `EE.SRCI.2Q`**
  - _Syntax:_ `EE.SLCI.2Q qx, qy, imm`
  - _Execution:_ Concatenates `qx` and `qy`, shifts left/right by `(imm + 1) * 8` bits, and splits back into the two `QR` registers.
- **`EE.SLCXXP.2Q` / `EE.SRCXXP.2Q`**
  - _Syntax:_ `EE.SLCXXP.2Q qx, qy, as, at`
  - _Execution:_ Same as above, but the shift amount is dictated by the lower 4 bits of `as`. Also updates `as` by adding `at`.
- **`EE.VSR.32` / `EE.VSL.32`**
  - _Syntax:_ `EE.VSR.32 qz, qx`
  - _Execution:_ Arithmetic right/left shift on independent 32-bit lanes inside `qx`, target `qz`. Shift amount is dictated by `SAR`.

### 3.6 FFT Specific Macros

- **`EE.BITREV`**
  - _Syntax:_ `EE.BITREV qz, as, imm`
  - _Execution:_ Reads a value from `as`, bit-reverses it based on the `FFT_BIT_WIDTH` special register, compares the original and reversed value, takes the maximum, zero-pads to 16-bits, and stores it in `qz` (lane specified by `imm`).
- **`EE.FFT.R2BF.S16{.ST.INCP}`**
  - _Syntax:_ `EE.FFT.R2BF.S16 qz, qx, qy, sel`
  - _Execution:_ Radix-2 Butterfly computation on 16-bit data. Generates dual outputs across two target `QR` registers based on a selector.
- **`EE.FFT.AMS.S16.[LD.INCP | LD.INCP.UAUP | LD.R32.DECP | ST.INCP]`**
  - _Syntax:_ `EE.FFT.AMS.S16.LD.INCP qz1, qz2, qx, qy, qm, as, sel2`
  - _Execution:_ Massive fused operation. Adds/subtracts two vectors `qx`/`qy`. Multiplies intermediate sums/differences by a third `qm` complex multiplier vector. Right-shifts everything by `SAR` to prevent explosion. Outputs real and imaginary parts to two different targets `qz1`/`qz2`.
  - _Quirk (.LD.R32.DECP):_ Fetches memory in **big-endian 32-bit word order** and _decrements_ the `as` pointer by 16.
  - _Quirk (.LD.INCP.UAUP):_ Splices fetched memory with the `UA_STATE` cache register, shifted by `SAR_BYTE`, effectively performing FFT math natively on an unaligned sliding window.
- **`EE.FFT.VST.R32.DECP`**
  - _Syntax:_ `EE.FFT.VST.R32.DECP qx, as, imm`
  - _Execution:_ Stores 16-bytes from `qx` to `as` in big-endian word order, arithmetic right-shifting by 0 or 1 based on `imm`, and decrements the pointer by 16.

### 3.7 CPU Fast GPIO

- **`EE.SET_BIT_GPIO_OUT` / `EE.CLR_BIT_GPIO_OUT`**
  - _Syntax:_ `EE.SET_BIT_GPIO_OUT imm`
  - _Execution:_ Bypasses standard memory-mapped IO. Takes an 8-bit immediate mask and directly drives physical voltage on the CPU's dedicated fast GPIO interface in a single cycle. Pins must be pre-mapped in the GPIO matrix to `pro_alonegpio_out[0..7]`.

---

## 4. PIE Register Map Reference

Below is the complete map of registers available to and utilized by the ESP32-S3 Processor Instruction Extensions (PIE).

| Category                   | Register(s)        | Width        | Description / Primary Usage                                                                                                                   |
| :------------------------- | :----------------- | :----------- | :-------------------------------------------------------------------------------------------------------------------------------------------- |
| **Vector Data (QR)**       | `q0` – `q7`        | 128-bit      | The main SIMD general-purpose vector registers. Can be treated as containing 8/16/32-bit discrete lanes or massive 128-bit blocks.            |
| **General Purpose (AR)**   | `a0` – `a15`       | 32-bit       | Standard Xtensa Core registers. Used by PIE for storing memory addresses, index offsets, scalar parameters, and immediate shifts.             |
| **Global Accumulator**     | `ACCX`             | 40-bit       | Single specialized scalar accumulator. Takes the collapsed sum of all parallel vector multiplications (e.g., `VMULAS.*.ACCX`).                |
| **Segmented Accumulators** | `QACC_H`, `QACC_L` | 160-bit (x2) | Parallel lane accumulators holding 320 bits of total state. Divided into 20-bit lanes (for 8-bit ops) or 40-bit lanes (for 16-bit ops).       |
| **Arithmetic Shift**       | `SAR`              | 6-bit        | (Special Register) Determines the automatic arithmetic right-shift scaling applied during multiplications and FFT ops to prevent overflow.    |
| **Unaligned Byte Shift**   | `SAR_BYTE`         | 4-bit        | (User Register) Stores a 0-15 byte offset used explicitly by `EE.SRC.Q` instructions to realign memory vectors read from unaligned addresses. |
| **FFT Width Control**      | `FFT_BIT_WIDTH`    | 4-bit        | (User Register) Defines the active span (3 to 10 bits) for the FFT Bit-Reversal instructions (`EE.BITREV`).                                   |
| **Unaligned State Cache**  | `UA_STATE`         | 128-bit      | (User Register) A dedicated buffer utilized exclusively by `EE.FFT.AMS.*.UAUP` to carry over the trailing bytes of a previous memory fetch.   |
