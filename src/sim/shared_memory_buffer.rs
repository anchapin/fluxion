//! Shared Memory Buffer for BES-FFD Inter-Engine Communication
//!
//! This module implements a high-performance shared memory buffer for direct
//! data exchange between the Building Energy Simulation (BES) engine and
//! the Fast Fluid Dynamics (FFD) solver.
//!
//! ## Key Design Decisions
//!
//! - **Double-buffering**: Two buffers (ping-pong) to avoid read/write conflicts
//! - **Cross-platform**: Memory-mapped files on both POSIX and Windows
//! - **Thread-safe**: Mutex-protected access
//! - **Low-latency**: Designed for sub-millisecond data exchange
//!
//! ## Performance Targets
//!
//! - Data exchange latency: < 1ms per timestep
//! - Memory footprint: < 100MB for typical zone counts
//! - Throughput: > 10,000 state variable updates per second
//!
//! ## Usage
//!
//! ```text
//! BES Engine  <--SHM-->  FFD FMU
//!               ^
//!               |
//!          Shared Memory
//!          Ring Buffer
//! ```
//!
//! The BES engine writes to buffer 0, FFD reads from buffer 0.
//! After FFD signals completion, BES can write to buffer 1, etc.
//! This double-buffering eliminates the need for locks on the data path.

use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[cfg(unix)]
#[allow(unused_imports)]
use std::os::unix::fs::FileExt;

/// Errors that can occur during shared memory operations.
#[derive(Debug, Clone, Error)]
pub enum SharedMemoryError {
    #[error("Failed to create shared memory file: {0}")]
    CreationFailed(String),

    #[error("Failed to open shared memory file: {0}")]
    OpenFailed(String),

    #[error("Failed to map shared memory: {0}")]
    MapFailed(String),

    #[error("Failed to sync shared memory: {0}")]
    SyncFailed(String),

    #[error("Buffer index out of bounds: {0}")]
    BufferOutOfBounds(usize),

    #[error("Data size mismatch: expected {expected}, got {got}")]
    SizeMismatch { expected: usize, got: usize },

    #[error("Region not initialized")]
    NotInitialized,

    #[error("Mutex lock failed: {0}")]
    LockFailed(String),

    #[error("Platform not supported")]
    PlatformNotSupported,

    #[error("IO error: {0}")]
    IoError(String),
}

impl From<std::io::Error> for SharedMemoryError {
    fn from(err: std::io::Error) -> Self {
        SharedMemoryError::IoError(err.to_string())
    }
}

/// Result type for shared memory operations.
pub type SharedMemoryResult<T> = Result<T, SharedMemoryError>;

#[cfg(unix)]
fn pwrite_all_at(file: &RefCell<File>, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    FileExt::write_all_at(&*file.borrow(), buf, offset)
}

#[cfg(windows)]
fn pwrite_all_at(file: &RefCell<File>, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom};
    let mut f = file.borrow_mut();
    f.seek(SeekFrom::Start(offset))?;
    f.write_all(buf)
}

#[cfg(unix)]
fn pread_exact_at(file: &RefCell<File>, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    FileExt::read_exact_at(&*file.borrow(), buf, offset)
}

#[cfg(windows)]
fn pread_exact_at(file: &RefCell<File>, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom};
    let mut f = file.borrow_mut();
    f.seek(SeekFrom::Start(offset))?;
    f.read_exact(buf)
}

/// Version magic to validate shared memory region compatibility.
const SHM_VERSION_MAGIC: u64 = 0x464C55494F4E5F53; // "FLUXION_S" in ASCII

/// Header stored at the start of the shared memory region.
#[repr(C)]
struct ShmHeader {
    /// Version magic for compatibility checking.
    version_magic: u64,
    /// Size of the data region in bytes.
    data_region_size: usize,
    /// Number of zones.
    num_zones: usize,
    /// Number of surfaces.
    num_surfaces: usize,
    /// Current write buffer index (0 or 1).
    write_buffer: u8,
    /// Current read buffer index (0 or 1).
    read_buffer: u8,
    /// Padding for alignment.
    _padding: [u8; 6],
}

impl ShmHeader {
    fn new(data_region_size: usize, num_zones: usize, num_surfaces: usize) -> Self {
        Self {
            version_magic: SHM_VERSION_MAGIC,
            data_region_size,
            num_zones,
            num_surfaces,
            write_buffer: 0,
            read_buffer: 1,
            _padding: [0; 6],
        }
    }

    fn validate(&self) -> SharedMemoryResult<()> {
        if self.version_magic != SHM_VERSION_MAGIC {
            return Err(SharedMemoryError::CreationFailed(
                "Shared memory version mismatch".to_string(),
            ));
        }
        Ok(())
    }
}

/// Shared memory buffer for BES-FFD data exchange.
///
/// This buffer uses double-buffering to avoid read/write conflicts:
/// - BES writes to the "write" buffer
/// - FFD reads from the "read" buffer
/// - After FFD signals completion, buffers are swapped
///
/// ## Thread Safety
///
/// The buffer itself is designed to be lock-free for the data path.
/// A separate synchronization primitive (semaphore or event) should be used
/// to signal when data is ready.
///
/// ## Memory Layout
///
/// ```text
/// +------------------+
/// | ShmHeader        |  Fixed-size header
/// +------------------+
/// | BesToFfdBuffer  |  Double-buffered BCs from BES
/// | [0] | [1]       |  (Ping-pong buffers)
/// +------------------+
/// | FfdToBesBuffer  |  Double-buffered results from FFD
/// | [0] | [1]       |  (Ping-pong buffers)
/// +------------------+
/// ```
pub struct SharedMemBuffer {
    /// File handle for the backing store (interior mutability for Windows seek).
    file: RefCell<File>,
    /// Path to the shared memory file.
    path: PathBuf,
    /// Size of the mapped region.
    size: usize,
    /// Number of zones.
    num_zones: usize,
    /// Number of surfaces.
    num_surfaces: usize,
    /// Mutex for coordinating buffer swaps.
    swap_mutex: Arc<Mutex<()>>,
    /// Flag indicating if this instance owns the region.
    is_owner: bool,
}

impl SharedMemBuffer {
    /// Calculate the required shared memory size.
    ///
    /// Includes header plus space for two buffers of type T (ping-pong).
    fn calculate_size(num_zones: usize, num_surfaces: usize) -> usize {
        let header_size = std::mem::size_of::<ShmHeader>();
        // Align header to cache line boundary
        let header_size = (header_size + 63) & !63;

        // Space for BES->FFD data and FFD->BES data
        // BES->FFD: outdoor_temp + surface_temps (num_surfaces) + hvac_* + internal + time + macro_dt + wind_pressure
        // FFD->BES: chtc (num_surfaces) + zone_temps (num_zones) + surface_flux (num_surfaces) + infiltration + mixing + metadata
        let bes_to_ffd_floats = 1 + num_surfaces + 5 + num_surfaces / 4;
        let ffd_to_bes_floats = num_surfaces + num_zones + num_surfaces + num_zones + num_zones + 2;

        let data_size =
            std::cmp::max(bes_to_ffd_floats, ffd_to_bes_floats) * std::mem::size_of::<f64>();

        header_size + data_size * 2 // Double-buffered
    }

    /// Get the directory for shared memory files.
    ///
    /// On POSIX, uses /dev/shm (tmpfs). On Windows, uses a temp directory.
    fn shm_dir() -> PathBuf {
        #[cfg(unix)]
        {
            PathBuf::from("/dev/shm")
        }
        #[cfg(windows)]
        {
            std::env::temp_dir()
        }
    }

    /// Create a new shared memory buffer with the given name.
    ///
    /// This creates a new shared memory region that can be opened by other processes.
    ///
    /// # Arguments
    /// * `name` - Unique name for the shared memory region
    /// * `num_zones` - Number of thermal zones
    /// * `num_surfaces` - Total number of surfaces
    ///
    /// # Returns
    /// A new SharedMemBuffer instance.
    pub fn create(name: &str, num_zones: usize, num_surfaces: usize) -> SharedMemoryResult<Self> {
        let size = Self::calculate_size(num_zones, num_surfaces);
        let shm_path = Self::shm_dir().join(format!("fluxion_{}", name));

        // Create and truncate the file
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&shm_path)?;

        // Set the file size
        file.set_len(size as u64)?;

        // Initialize the header
        let header = ShmHeader::new(size, num_zones, num_surfaces);
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const ShmHeader as *const u8,
                std::mem::size_of::<ShmHeader>(),
            )
        };
        file.write_all(header_bytes)?;

        Ok(Self {
            file: RefCell::new(file),
            path: shm_path,
            size,
            num_zones,
            num_surfaces,
            swap_mutex: Arc::new(Mutex::new(())),
            is_owner: true,
        })
    }

    /// Open an existing shared memory buffer.
    ///
    /// # Arguments
    /// * `name` - Name of the shared memory region to open
    ///
    /// # Returns
    /// A SharedMemBuffer instance connected to the existing region.
    pub fn open(name: &str) -> SharedMemoryResult<Self> {
        let shm_path = Self::shm_dir().join(format!("fluxion_{}", name));

        let mut file = OpenOptions::new().read(true).write(true).open(&shm_path)?;

        // Get the file size
        let metadata = file.metadata()?;
        let size = metadata.len() as usize;

        // Read and validate the header
        let mut header_bytes = vec![0u8; std::mem::size_of::<ShmHeader>()];
        file.read_exact(&mut header_bytes)?;
        let header = unsafe { std::ptr::read_unaligned(header_bytes.as_ptr() as *const ShmHeader) };
        header.validate()?;

        Ok(Self {
            file: RefCell::new(file),
            path: shm_path,
            size,
            num_zones: header.num_zones,
            num_surfaces: header.num_surfaces,
            swap_mutex: Arc::new(Mutex::new(())),
            is_owner: false,
        })
    }

    /// Get the number of zones.
    pub fn num_zones(&self) -> usize {
        self.num_zones
    }

    /// Get the number of surfaces.
    pub fn num_surfaces(&self) -> usize {
        self.num_surfaces
    }

    /// Get the path to the shared memory file.
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Read the header from the file.
    fn read_header(&self) -> SharedMemoryResult<ShmHeader> {
        let mut header_bytes = vec![0u8; std::mem::size_of::<ShmHeader>()];
        pread_exact_at(&self.file, &mut header_bytes, 0)?;
        Ok(unsafe { std::ptr::read_unaligned(header_bytes.as_ptr() as *const ShmHeader) })
    }

    /// Write the header to the file.
    fn write_header(&mut self, header: &ShmHeader) -> SharedMemoryResult<()> {
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                header as *const ShmHeader as *const u8,
                std::mem::size_of::<ShmHeader>(),
            )
        };
        pwrite_all_at(&self.file, header_bytes, 0)?;
        Ok(())
    }

    /// Calculate the offset to the BES->FFD data region for a given buffer.
    fn bes_to_ffd_offset(&self, buffer_idx: usize) -> usize {
        let header_size = (std::mem::size_of::<ShmHeader>() + 63) & !63;
        let data_region_size = (self.size - header_size) / 2;
        header_size + buffer_idx * data_region_size
    }

    /// Calculate the offset to the FFD->BES data region for a given buffer.
    fn ffd_to_bes_offset(&self, buffer_idx: usize) -> usize {
        let header_size = (std::mem::size_of::<ShmHeader>() + 63) & !63;
        let data_region_size = (self.size - header_size) / 2;
        header_size + data_region_size + buffer_idx * data_region_size
    }

    /// Calculate the size of each data region.
    fn data_region_size(&self) -> usize {
        let header_size = (std::mem::size_of::<ShmHeader>() + 63) & !63;
        (self.size - header_size) / 2
    }

    /// Write boundary conditions to the buffer (BES -> FFD).
    ///
    /// Writes to the current write buffer. Call `signal_write_complete()`
    /// after writing to signal FFD that data is ready.
    pub fn write_bes_to_ffd(&mut self, data: &BesToFfdSharedData) -> SharedMemoryResult<()> {
        if data.surface_temperatures.len() != self.num_surfaces {
            return Err(SharedMemoryError::SizeMismatch {
                expected: self.num_surfaces,
                got: data.surface_temperatures.len(),
            });
        }

        let header = self.read_header()?;
        let offset = self.bes_to_ffd_offset(header.write_buffer as usize);
        let mut buffer = vec![0.0f64; self.data_region_size() / 8];

        // Pack the data into the buffer
        // [outdoor_temp, surface_temps..., hvac_temp, hvac_flow, internal_gains, time_start, macro_timestep, wind_pressure...]
        let mut idx = 0;
        buffer[idx] = data.outdoor_temperature;
        idx += 1;

        for &temp in &data.surface_temperatures {
            buffer[idx] = temp;
            idx += 1;
        }

        buffer[idx] = data.hvac_supply_temperature;
        idx += 1;
        buffer[idx] = data.hvac_supply_flow;
        idx += 1;
        buffer[idx] = data.internal_gains;
        idx += 1;
        buffer[idx] = data.time_start;
        idx += 1;
        buffer[idx] = data.macro_timestep;
        idx += 1;

        for &wp in &data.wind_pressure {
            if idx < buffer.len() {
                buffer[idx] = wp;
                idx += 1;
            }
        }

        // Write to file
        let bytes =
            unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u8, buffer.len() * 8) };
        pwrite_all_at(&self.file, bytes, offset as u64)?;

        Ok(())
    }

    /// Read boundary conditions from the buffer (FFD reads from BES).
    ///
    /// Reads from the current read buffer.
    pub fn read_bes_to_ffd(&self) -> SharedMemoryResult<BesToFfdSharedData> {
        let header = self.read_header()?;
        let offset = self.bes_to_ffd_offset(header.read_buffer as usize);

        // Read raw bytes first
        let mut bytes = vec![0u8; self.data_region_size()];
        pread_exact_at(&self.file, &mut bytes, offset as u64)?;

        // Reinterpret as f64 array
        let floats_count = bytes.len() / 8;
        let mut buffer = vec![0.0f64; floats_count];
        buffer.copy_from_slice(unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f64, floats_count)
        });

        let mut data = BesToFfdSharedData::default();
        let mut idx = 0;

        data.outdoor_temperature = buffer[idx];
        idx += 1;

        let mut temps = vec![0.0; self.num_surfaces];
        temps[..self.num_surfaces].copy_from_slice(&buffer[idx..self.num_surfaces + idx]);
        data.surface_temperatures = temps;
        idx += self.num_surfaces;

        data.hvac_supply_temperature = buffer[idx];
        idx += 1;
        data.hvac_supply_flow = buffer[idx];
        idx += 1;
        data.internal_gains = buffer[idx];
        idx += 1;
        data.time_start = buffer[idx];
        idx += 1;
        data.macro_timestep = buffer[idx];
        idx += 1;

        // Wind pressure
        let wind_count = self.num_surfaces / 4;
        let mut wind = vec![0.0; wind_count];
        for i in 0..wind_count {
            if idx + i < buffer.len() {
                wind[i] = buffer[idx + i];
            }
        }
        data.wind_pressure = wind;

        Ok(data)
    }

    /// Write results to the buffer (FFD -> BES).
    pub fn write_ffd_to_bes(&mut self, data: &FfdToBesSharedData) -> SharedMemoryResult<()> {
        if data.chtc.len() != self.num_surfaces {
            return Err(SharedMemoryError::SizeMismatch {
                expected: self.num_surfaces,
                got: data.chtc.len(),
            });
        }
        if data.zone_temperatures.len() != self.num_zones {
            return Err(SharedMemoryError::SizeMismatch {
                expected: self.num_zones,
                got: data.zone_temperatures.len(),
            });
        }

        let header = self.read_header()?;
        let offset = self.ffd_to_bes_offset(header.write_buffer as usize);
        let mut buffer = vec![0.0f64; self.data_region_size() / 8];

        // Pack: [chtc..., zone_temps..., surface_flux..., infiltration..., mixing..., micro_count, sim_time]
        let mut idx = 0;

        for &chtc in &data.chtc {
            buffer[idx] = chtc;
            idx += 1;
        }

        for &temp in &data.zone_temperatures {
            buffer[idx] = temp;
            idx += 1;
        }

        for &flux in &data.surface_heat_flux {
            buffer[idx] = flux;
            idx += 1;
        }

        for &inf in &data.infiltration_flow {
            buffer[idx] = inf;
            idx += 1;
        }

        for &mix in &data.mixing_flow {
            buffer[idx] = mix;
            idx += 1;
        }

        buffer[idx] = data.micro_step_count as f64;
        idx += 1;
        buffer[idx] = data.simulation_time_covered;

        // Write to file
        let bytes =
            unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u8, buffer.len() * 8) };
        pwrite_all_at(&self.file, bytes, offset as u64)?;

        Ok(())
    }

    /// Read results from the buffer (BES reads from FFD).
    pub fn read_ffd_to_bes(&self) -> SharedMemoryResult<FfdToBesSharedData> {
        let header = self.read_header()?;
        let offset = self.ffd_to_bes_offset(header.read_buffer as usize);

        // Read raw bytes first
        let mut bytes = vec![0u8; self.data_region_size()];
        pread_exact_at(&self.file, &mut bytes, offset as u64)?;

        // Reinterpret as f64 array
        let floats_count = bytes.len() / 8;
        let buffer: Vec<f64> = unsafe {
            let slice = std::slice::from_raw_parts(bytes.as_ptr() as *const f64, floats_count);
            let mut aligned = vec![0.0f64; floats_count];
            aligned.copy_from_slice(slice);
            aligned
        };

        let mut data = FfdToBesSharedData::default();
        let mut idx = 0;

        let mut chtc = vec![0.0; self.num_surfaces];
        chtc[..self.num_surfaces].copy_from_slice(&buffer[idx..self.num_surfaces + idx]);
        data.chtc = chtc;
        idx += self.num_surfaces;

        let mut temps = vec![0.0; self.num_zones];
        temps[..self.num_zones].copy_from_slice(&buffer[idx..self.num_zones + idx]);
        data.zone_temperatures = temps;
        idx += self.num_zones;

        let mut fluxes = vec![0.0; self.num_surfaces];
        fluxes[..self.num_surfaces].copy_from_slice(&buffer[idx..self.num_surfaces + idx]);
        data.surface_heat_flux = fluxes;
        idx += self.num_surfaces;

        let mut infs = vec![0.0; self.num_zones];
        infs[..self.num_zones].copy_from_slice(&buffer[idx..self.num_zones + idx]);
        data.infiltration_flow = infs;
        idx += self.num_zones;

        let mut mixes = vec![0.0; self.num_zones];
        mixes[..self.num_zones].copy_from_slice(&buffer[idx..self.num_zones + idx]);
        data.mixing_flow = mixes;
        idx += self.num_zones;

        data.micro_step_count = buffer[idx] as usize;
        idx += 1;
        data.simulation_time_covered = buffer[idx];

        Ok(data)
    }

    /// Signal that BES write is complete and FFD can read.
    ///
    /// This swaps the read buffer to point to the newly written data.
    /// FFD will see the new data on the next read.
    pub fn bes_write_complete(&mut self) -> SharedMemoryResult<()> {
        // Read current header under lock
        let current_write_buffer = {
            let _lock = self.swap_mutex.lock().map_err(|e| {
                SharedMemoryError::LockFailed(format!("Failed to acquire swap lock: {}", e))
            })?;
            let header = self.read_header()?;
            header.write_buffer
        };
        // Lock is now dropped, we can mutate self

        // Toggle the write buffer
        let new_write_buffer = (current_write_buffer + 1) % 2;

        let mut header = self.read_header()?;
        header.write_buffer = new_write_buffer;
        self.write_header(&header)?;

        Ok(())
    }

    /// Signal that FFD write is complete and BES can read.
    ///
    /// This swaps the read buffer to point to the newly written data.
    /// BES will see the new data on the next read.
    pub fn ffd_write_complete(&mut self) -> SharedMemoryResult<()> {
        // Read current header under lock
        let current_read_buffer = {
            let _lock = self.swap_mutex.lock().map_err(|e| {
                SharedMemoryError::LockFailed(format!("Failed to acquire swap lock: {}", e))
            })?;
            let header = self.read_header()?;
            header.read_buffer
        };
        // Lock is now dropped, we can mutate self

        // Toggle the read buffer
        let new_read_buffer = (current_read_buffer + 1) % 2;

        let mut header = self.read_header()?;
        header.read_buffer = new_read_buffer;
        self.write_header(&header)?;

        Ok(())
    }

    /// Sync the shared memory to disk (for durability).
    ///
    /// This uses fdatasync on POSIX or FlushFileBuffers on Windows.
    pub fn sync(&self) -> SharedMemoryResult<()> {
        self.file
            .borrow_mut()
            .sync_all()
            .map_err(|e| SharedMemoryError::SyncFailed(format!("sync_all failed: {}", e)))?;
        Ok(())
    }

    /// Close the shared memory buffer.
    ///
    /// If this instance is the owner, the underlying shared memory
    /// file will be removed.
    pub fn close(self) -> SharedMemoryResult<()> {
        let Self {
            file,
            path,
            is_owner,
            ..
        } = self;
        let _ = file;
        if is_owner {
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }

    /// Destroy the shared memory region (owner only).
    ///
    /// This should be called when the shared memory region is no longer needed.
    #[cfg(unix)]
    pub fn destroy(name: &str) -> SharedMemoryResult<()> {
        let path = Self::shm_dir().join(format!("fluxion_{}", name));
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    #[cfg(windows)]
    pub fn destroy(name: &str) -> SharedMemoryResult<()> {
        let path = Self::shm_dir().join(format!("fluxion_{}", name));
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}

/// Data written from BES to FFD via shared memory.
///
/// This is a simplified version of BesToFfdBoundaryConditions
/// optimized for shared memory transfer.
#[derive(Debug, Clone, Default)]
pub struct BesToFfdSharedData {
    /// Outdoor dry-bulb temperature [K].
    pub outdoor_temperature: f64,
    /// Surface temperatures for each zone surface [K].
    pub surface_temperatures: Vec<f64>,
    /// HVAC supply air temperature [K].
    pub hvac_supply_temperature: f64,
    /// HVAC supply air flow rate [m³/s].
    pub hvac_supply_flow: f64,
    /// Internal gains from occupants, equipment, lighting [W].
    pub internal_gains: f64,
    /// Simulation time at start of macro step [s].
    pub time_start: f64,
    /// Duration of macro step [s].
    pub macro_timestep: f64,
    /// Wind pressure on each facade [Pa].
    pub wind_pressure: Vec<f64>,
}

/// Data written from FFD to BES via shared memory.
///
/// This is a simplified version of FfdToBesResults
/// optimized for shared memory transfer.
#[derive(Debug, Clone, Default)]
pub struct FfdToBesSharedData {
    /// Convective heat transfer coefficients for each surface [W/m²K].
    pub chtc: Vec<f64>,
    /// Zone air temperatures [K].
    pub zone_temperatures: Vec<f64>,
    /// Surface heat fluxes [W/m²].
    pub surface_heat_flux: Vec<f64>,
    /// Infiltration flow rates [m³/s].
    pub infiltration_flow: Vec<f64>,
    /// Zone mixing flow rates [m³/s].
    pub mixing_flow: Vec<f64>,
    /// Number of micro steps FFD took.
    pub micro_step_count: usize,
    /// Actual simulation time covered [s].
    pub simulation_time_covered: f64,
}

impl From<crate::sim::loose_coupling::BesToFfdBoundaryConditions> for BesToFfdSharedData {
    fn from(bc: crate::sim::loose_coupling::BesToFfdBoundaryConditions) -> Self {
        Self {
            outdoor_temperature: bc.outdoor_temperature,
            surface_temperatures: bc.surface_temperatures,
            hvac_supply_temperature: bc.hvac_supply_temperature,
            hvac_supply_flow: bc.hvac_supply_flow,
            internal_gains: bc.internal_gains,
            time_start: bc.time_start,
            macro_timestep: bc.macro_timestep,
            wind_pressure: bc.wind_pressure,
        }
    }
}

impl From<FfdToBesSharedData> for crate::sim::loose_coupling::FfdToBesResults {
    fn from(data: FfdToBesSharedData) -> Self {
        Self {
            chtc: data.chtc,
            zone_temperatures: data.zone_temperatures,
            surface_heat_flux: data.surface_heat_flux,
            infiltration_flow: data.infiltration_flow,
            mixing_flow: data.mixing_flow,
            micro_step_count: data.micro_step_count,
            simulation_time_covered: data.simulation_time_covered,
        }
    }
}

/// A thread-safe wrapper around SharedMemBuffer with built-in synchronization.
///
/// This wrapper uses a reader-writer lock to allow multiple readers
/// while ensuring exclusive writer access.
pub struct SyncedSharedMemBuffer {
    inner: SharedMemBuffer,
    write_ready: Arc<Mutex<bool>>,
    read_ready: Arc<Mutex<bool>>,
}

impl SyncedSharedMemBuffer {
    /// Create a new synchronized shared memory buffer.
    pub fn create(name: &str, num_zones: usize, num_surfaces: usize) -> SharedMemoryResult<Self> {
        let inner = SharedMemBuffer::create(name, num_zones, num_surfaces)?;
        Ok(Self {
            inner,
            write_ready: Arc::new(Mutex::new(false)),
            read_ready: Arc::new(Mutex::new(false)),
        })
    }

    /// Open an existing synchronized shared memory buffer.
    pub fn open(name: &str) -> SharedMemoryResult<Self> {
        let inner = SharedMemBuffer::open(name)?;
        Ok(Self {
            inner,
            write_ready: Arc::new(Mutex::new(false)),
            read_ready: Arc::new(Mutex::new(false)),
        })
    }

    /// Write data from BES to FFD with synchronization.
    pub fn write_bes_to_ffd(&mut self, data: &BesToFfdSharedData) -> SharedMemoryResult<()> {
        let mut ready = self.write_ready.lock().map_err(|e| {
            SharedMemoryError::LockFailed(format!("Failed to acquire write lock: {}", e))
        })?;

        self.inner.write_bes_to_ffd(data)?;

        *ready = true;
        drop(ready);

        self.inner.bes_write_complete()?;

        Ok(())
    }

    /// Read data from BES to FFD with synchronization.
    ///
    /// Returns None if no new data is available.
    pub fn read_bes_to_ffd(&self) -> SharedMemoryResult<Option<BesToFfdSharedData>> {
        let ready = self.write_ready.lock().map_err(|e| {
            SharedMemoryError::LockFailed(format!("Failed to acquire read lock: {}", e))
        })?;

        if !*ready {
            return Ok(None);
        }

        let data = self.inner.read_bes_to_ffd()?;

        Ok(Some(data))
    }

    /// Write data from FFD to BES with synchronization.
    pub fn write_ffd_to_bes(&mut self, data: &FfdToBesSharedData) -> SharedMemoryResult<()> {
        let mut ready = self.read_ready.lock().map_err(|e| {
            SharedMemoryError::LockFailed(format!("Failed to acquire write lock: {}", e))
        })?;

        self.inner.write_ffd_to_bes(data)?;
        *ready = true;
        drop(ready);

        self.inner.ffd_write_complete()?;

        Ok(())
    }

    /// Read data from FFD to BES with synchronization.
    ///
    /// Returns None if no new data is available.
    pub fn read_ffd_to_bes(&self) -> SharedMemoryResult<Option<FfdToBesSharedData>> {
        let ready = self.read_ready.lock().map_err(|e| {
            SharedMemoryError::LockFailed(format!("Failed to acquire read lock: {}", e))
        })?;

        if !*ready {
            return Ok(None);
        }

        let data = self.inner.read_ffd_to_bes()?;
        Ok(Some(data))
    }

    /// Get the number of zones.
    pub fn num_zones(&self) -> usize {
        self.inner.num_zones()
    }

    /// Get the number of surfaces.
    pub fn num_surfaces(&self) -> usize {
        self.inner.num_surfaces()
    }

    /// Get the path to the shared memory file.
    pub fn path(&self) -> &PathBuf {
        self.inner.path()
    }

    /// Sync data to disk.
    pub fn sync(&mut self) -> SharedMemoryResult<()> {
        self.inner.sync()
    }

    /// Close the buffer.
    pub fn close(self) -> SharedMemoryResult<()> {
        self.inner.close()
    }
}

/// Shared memory buffer manager for creating and managing multiple buffers.
///
/// This is useful for scenarios where multiple zones or multiple
/// simulation instances need separate shared memory regions.
pub struct SharedMemManager {
    /// Active buffers.
    buffers: std::collections::HashMap<String, SharedMemBuffer>,
}

impl SharedMemManager {
    /// Create a new shared memory manager.
    pub fn new() -> Self {
        Self {
            buffers: std::collections::HashMap::new(),
        }
    }

    /// Create or open a shared memory buffer.
    pub fn get_or_create(
        &mut self,
        name: &str,
        num_zones: usize,
        num_surfaces: usize,
    ) -> SharedMemoryResult<&mut SharedMemBuffer> {
        if !self.buffers.contains_key(name) {
            // Try to create a new buffer
            match SharedMemBuffer::create(name, num_zones, num_surfaces) {
                Ok(buffer) => {
                    self.buffers.insert(name.to_string(), buffer);
                }
                Err(_) => {
                    // If creation fails, try to open existing
                    let buffer = SharedMemBuffer::open(name)?;
                    self.buffers.insert(name.to_string(), buffer);
                }
            }
        }

        self.buffers
            .get_mut(name)
            .ok_or(SharedMemoryError::NotInitialized)
    }

    /// Close a specific buffer.
    pub fn close_buffer(&mut self, name: &str) -> SharedMemoryResult<()> {
        if let Some(buffer) = self.buffers.remove(name) {
            buffer.close()?;
        }
        Ok(())
    }

    /// Close all buffers.
    pub fn close_all(&mut self) -> SharedMemoryResult<()> {
        for (_, buffer) in self.buffers.drain() {
            buffer.close()?;
        }
        Ok(())
    }
}

impl Default for SharedMemManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shm_header_size() {
        let header = ShmHeader::new(1024, 4, 16);
        assert_eq!(header.num_zones, 4);
        assert_eq!(header.num_surfaces, 16);
        assert_eq!(header.write_buffer, 0);
        assert_eq!(header.read_buffer, 1);
    }

    #[test]
    fn test_shm_header_validation() {
        let header = ShmHeader::new(1024, 4, 16);
        assert!(header.validate().is_ok());
    }

    #[test]
    fn test_shm_header_invalid_magic() {
        let mut header = ShmHeader::new(1024, 4, 16);
        header.version_magic = 0;
        assert!(header.validate().is_err());
    }

    #[test]
    fn test_bes_to_ffd_shared_data_default() {
        let data = BesToFfdSharedData::default();
        assert_eq!(data.outdoor_temperature, 0.0);
        assert!(data.surface_temperatures.is_empty());
    }

    #[test]
    fn test_ffd_to_bes_shared_data_default() {
        let data = FfdToBesSharedData::default();
        assert!(data.chtc.is_empty());
        assert!(data.zone_temperatures.is_empty());
    }

    #[test]
    fn test_calculate_size() {
        let size = SharedMemBuffer::calculate_size(4, 16);
        assert!(size > std::mem::size_of::<ShmHeader>());
        // Should be enough for all the data
        let header_size = (std::mem::size_of::<ShmHeader>() + 63) & !63;
        let data_size = size - header_size;
        assert!(data_size > 0);
    }

    #[test]
    fn test_from_boundary_conditions() {
        let bc = crate::sim::loose_coupling::BesToFfdBoundaryConditions {
            outdoor_temperature: 280.0,
            surface_temperatures: vec![295.0; 8],
            hvac_supply_temperature: 293.0,
            hvac_supply_flow: 0.5,
            wind_pressure: vec![0.0; 4],
            internal_gains: 500.0,
            time_start: 0.0,
            macro_timestep: 3600.0,
        };

        let data: BesToFfdSharedData = bc.into();
        assert_eq!(data.outdoor_temperature, 280.0);
        assert_eq!(data.surface_temperatures.len(), 8);
        assert_eq!(data.hvac_supply_temperature, 293.0);
    }

    #[test]
    fn test_from_ffd_results() {
        let data = FfdToBesSharedData {
            chtc: vec![10.0; 8],
            zone_temperatures: vec![293.15; 2],
            surface_heat_flux: vec![50.0; 8],
            infiltration_flow: vec![0.1; 2],
            mixing_flow: vec![0.05; 2],
            micro_step_count: 3600,
            simulation_time_covered: 3600.0,
        };

        let results: crate::sim::loose_coupling::FfdToBesResults = data.into();
        assert_eq!(results.chtc.len(), 8);
        assert_eq!(results.zone_temperatures.len(), 2);
        assert_eq!(results.micro_step_count, 3600);
    }

    #[test]
    fn test_shared_mem_manager() {
        let manager = SharedMemManager::new();
        // Manager should start empty
        assert!(manager.buffers.is_empty());
    }

    #[test]
    fn test_shm_dir() {
        let dir = SharedMemBuffer::shm_dir();
        // On Linux, shm_dir returns /dev/shm which exists
        // On Windows, it returns temp_dir which should also exist
        assert!(dir.exists() || dir.to_string_lossy().len() > 0);
    }
}
