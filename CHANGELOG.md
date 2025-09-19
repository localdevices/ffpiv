## [0.1.4] - 2025-09-19
### Added
### Changed
- A new flag `clip_norm` is added to `api.piv`, `api.piv_stack` and all downstream functions to allow a user to decide 
  if intensities of interrogation windows should be clipped between 0 and the maximum value of the interrogation window.
  If preprocessing is applied by external functions, the user should typically not clip the intensities.
  `clip_norm=True` makes all results consistent with the original behaviour and with OpenPIV.

### Deprecated
### Removed
### Fixed
### Security


## [0.1.3] - 2025-02-25
### Added
### Changed
### Deprecated
### Removed
### Fixed
* one numba function was not cached, this is now fixed.

### Security

## [0.1.2] - 2024-12-10
### Added
### Changed
* `api.cross_corr` now only reports possible memory issues when `verbose=True`

### Deprecated
### Removed
### Fixed
### Security


## [0.1.1] - 2024-12-01
### Added
* Additional documentation for FF-PIV API
* New examples in the FF-PIV package

### Changed
* Improved performance of FF-PIV when processing large datasets

### Deprecated
### Removed
### Fixed
* Bug fixes for data processing errors

### Security

## [0.1.0] - 2024-11-14
### Added
* First release of FF-PIV

### Changed
### Deprecated
### Removed
### Fixed
### Security
