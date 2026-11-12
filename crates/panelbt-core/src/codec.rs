//! Minimal little-endian binary encoding for simulator snapshots.
//!
//! The format is private to this crate version: snapshots carry a magic
//! string and a format number and are rejected on mismatch.

pub trait Codec: Sized {
    fn put(&self, w: &mut Vec<u8>);
    fn take(r: &mut &[u8]) -> Result<Self, String>;
}

fn bytes<const K: usize>(r: &mut &[u8]) -> Result<[u8; K], String> {
    if r.len() < K {
        return Err("snapshot: truncated".into());
    }
    let (head, tail) = r.split_at(K);
    *r = tail;
    Ok(head.try_into().unwrap())
}

impl Codec for u64 {
    fn put(&self, w: &mut Vec<u8>) {
        w.extend_from_slice(&self.to_le_bytes());
    }
    fn take(r: &mut &[u8]) -> Result<Self, String> {
        Ok(u64::from_le_bytes(bytes::<8>(r)?))
    }
}
impl Codec for i64 {
    fn put(&self, w: &mut Vec<u8>) {
        w.extend_from_slice(&self.to_le_bytes());
    }
    fn take(r: &mut &[u8]) -> Result<Self, String> {
        Ok(i64::from_le_bytes(bytes::<8>(r)?))
    }
}
impl Codec for f64 {
    fn put(&self, w: &mut Vec<u8>) {
        w.extend_from_slice(&self.to_bits().to_le_bytes());
    }
    fn take(r: &mut &[u8]) -> Result<Self, String> {
        Ok(f64::from_bits(u64::from_le_bytes(bytes::<8>(r)?)))
    }
}
impl Codec for usize {
    fn put(&self, w: &mut Vec<u8>) {
        (*self as u64).put(w)
    }
    fn take(r: &mut &[u8]) -> Result<Self, String> {
        usize::try_from(u64::take(r)?).map_err(|_| "snapshot: index overflow".into())
    }
}
impl Codec for bool {
    fn put(&self, w: &mut Vec<u8>) {
        w.push(u8::from(*self));
    }
    fn take(r: &mut &[u8]) -> Result<Self, String> {
        match bytes::<1>(r)?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err("snapshot: invalid boolean".into()),
        }
    }
}
impl Codec for u8 {
    fn put(&self, w: &mut Vec<u8>) {
        w.push(*self);
    }
    fn take(r: &mut &[u8]) -> Result<Self, String> {
        Ok(bytes::<1>(r)?[0])
    }
}
impl<T: Codec> Codec for Vec<T> {
    fn put(&self, w: &mut Vec<u8>) {
        self.len().put(w);
        for x in self {
            x.put(w);
        }
    }
    fn take(r: &mut &[u8]) -> Result<Self, String> {
        let k = usize::take(r)?;
        // Every element takes at least one byte: reject impossible lengths early.
        if k > r.len() {
            return Err("snapshot: truncated".into());
        }
        (0..k).map(|_| T::take(r)).collect()
    }
}
impl<T: Codec> Codec for Option<T> {
    fn put(&self, w: &mut Vec<u8>) {
        self.is_some().put(w);
        if let Some(x) = self {
            x.put(w);
        }
    }
    fn take(r: &mut &[u8]) -> Result<Self, String> {
        Ok(if bool::take(r)? {
            Some(T::take(r)?)
        } else {
            None
        })
    }
}
impl<const K: usize> Codec for [u64; K] {
    fn put(&self, w: &mut Vec<u8>) {
        for x in self {
            x.put(w);
        }
    }
    fn take(r: &mut &[u8]) -> Result<Self, String> {
        let mut a = [0; K];
        for x in &mut a {
            *x = u64::take(r)?;
        }
        Ok(a)
    }
}
macro_rules! tuple {
    ($($name:ident),+) => {
        impl<$($name: Codec),+> Codec for ($($name,)+) {
            #[allow(non_snake_case)]
            fn put(&self, w: &mut Vec<u8>) {
                let ($($name,)+) = self;
                $($name.put(w);)+
            }
            fn take(r: &mut &[u8]) -> Result<Self, String> {
                Ok(($($name::take(r)?,)+))
            }
        }
    };
}
tuple!(A, B);
tuple!(A, B, C);
tuple!(A, B, C, D);
tuple!(A, B, C, D, E);
tuple!(A, B, C, D, E, F, G, H, I);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_truncation() {
        let value: (Vec<(usize, f64)>, Option<Vec<i64>>, bool) =
            (vec![(3, -0.0), (7, f64::NAN)], Some(vec![-1, 5]), true);
        let mut w = Vec::new();
        value.put(&mut w);
        let mut r = w.as_slice();
        let back = <(Vec<(usize, f64)>, Option<Vec<i64>>, bool)>::take(&mut r).unwrap();
        assert!(r.is_empty());
        assert_eq!(back.0[0].0, 3);
        assert_eq!(back.0[0].1.to_bits(), (-0.0f64).to_bits());
        assert!(back.0[1].1.is_nan());
        assert_eq!(back.1, Some(vec![-1, 5]));
        let mut short = &w[..w.len() - 1];
        assert!(<(Vec<(usize, f64)>, Option<Vec<i64>>, bool)>::take(&mut short).is_err());
    }
}
