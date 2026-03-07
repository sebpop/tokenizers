use pcre2_sys::*;
use std::cell::RefCell;
use std::error::Error;
use std::ffi::CString;
use std::ptr;

pub struct SysRegex {
    code: *mut pcre2_code_8,
}

unsafe impl Send for SysRegex {}
unsafe impl Sync for SysRegex {}

impl std::fmt::Debug for SysRegex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SysRegex").finish()
    }
}

impl Drop for SysRegex {
    fn drop(&mut self) {
        unsafe { pcre2_code_free_8(self.code) };
    }
}

struct ThreadMatchData {
    md: *mut pcre2_match_data_8,
}

impl Drop for ThreadMatchData {
    fn drop(&mut self) {
        if !self.md.is_null() {
            unsafe { pcre2_match_data_free_8(self.md) };
        }
    }
}

thread_local! {
    static MATCH_DATA: RefCell<ThreadMatchData> = RefCell::new(ThreadMatchData { md: ptr::null_mut() });
}

impl SysRegex {
    pub fn new(
        regex_str: &str,
    ) -> std::result::Result<Self, Box<dyn Error + Send + Sync + 'static>> {
        let pattern = CString::new(regex_str)?;
        let mut error_code: i32 = 0;
        let mut error_offset: usize = 0;

        let code = unsafe {
            pcre2_compile_8(
                pattern.as_ptr() as *const u8,
                regex_str.len(),
                PCRE2_UTF | PCRE2_UCP,
                &mut error_code,
                &mut error_offset,
                ptr::null_mut(),
            )
        };
        if code.is_null() {
            return Err(format!(
                "pcre2_compile failed at offset {error_offset} with error {error_code}"
            )
            .into());
        }

        let jit_rc = unsafe { pcre2_jit_compile_8(code, PCRE2_JIT_COMPLETE) };
        if jit_rc != 0 {
            unsafe { pcre2_code_free_8(code) };
            return Err(format!("pcre2_jit_compile failed with error {jit_rc}").into());
        }

        Ok(Self { code })
    }

    pub fn find_iter<'r, 't>(&'r self, inside: &'t str) -> Matches<'r, 't> {
        Matches {
            regex: self,
            text: inside.as_bytes(),
            last: 0,
        }
    }

    fn find_at(&self, text: &[u8], start: usize) -> Option<(usize, usize)> {
        MATCH_DATA.with(|cell| {
            let mut tmd = cell.borrow_mut();
            if tmd.md.is_null() {
                tmd.md = unsafe {
                    pcre2_match_data_create_from_pattern_8(self.code, ptr::null_mut())
                };
            }

            let rc = unsafe {
                pcre2_jit_match_8(
                    self.code,
                    text.as_ptr(),
                    text.len(),
                    start,
                    0,
                    tmd.md,
                    ptr::null_mut(),
                )
            };

            if rc < 1 {
                return None;
            }

            let ovector = unsafe { pcre2_get_ovector_pointer_8(tmd.md) };
            let match_start = unsafe { *ovector } as usize;
            let match_end = unsafe { *ovector.add(1) } as usize;
            Some((match_start, match_end))
        })
    }
}

pub struct Matches<'r, 't> {
    regex: &'r SysRegex,
    text: &'t [u8],
    last: usize,
}

impl Iterator for Matches<'_, '_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.last > self.text.len() {
            return None;
        }
        match self.regex.find_at(self.text, self.last) {
            Some((start, end)) => {
                if start == end {
                    self.last = end + 1;
                } else {
                    self.last = end;
                }
                Some((start, end))
            }
            None => None,
        }
    }
}
