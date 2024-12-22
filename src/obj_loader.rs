use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::str::{FromStr, SplitWhitespace};
use std::usize;

pub struct Index {
    pub vertex_index: usize,
    pub texcoord_index: usize,
    pub normal_index: usize,
}

pub struct Mesh {
    pub vertices: Vec<f32>,
    pub indices: Vec<Index>,
    pub tex_coords: Vec<f32>,
    pub normal_coords: Vec<f32>,
}

impl Index {
    fn new(vertex_index: usize, texcoord_index: usize, normal_index: usize) -> Self {
        Self {
            vertex_index,
            texcoord_index,
            normal_index,
        }
    }
}

pub struct ObjLoader {}

impl ObjLoader {
    fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
    where
        P: AsRef<Path>,
    {
        let file = File::open(filename)?;
        Ok(io::BufReader::new(file).lines())
    }

    fn err_string(lineno: usize) -> String {
        format!("parsing error on line {}", lineno)
    }

    fn next_item<'a, I, T>(iter: &mut I, lineno: usize) -> Result<T, Box<dyn std::error::Error>>
    where
        I: Iterator<Item = &'a str>,
        T: FromStr,
        <T as FromStr>::Err: std::error::Error,
        <T as FromStr>::Err: 'static,
    {
        let result = iter.next().ok_or(Self::err_string(lineno))?.parse::<T>()?;
        Result::Ok(result)
    }

    fn next_f32<'a, I>(iter: &mut I, lineno: usize) -> Result<f32, Box<dyn std::error::Error>>
    where
        I: Iterator<Item = &'a str>,
    {
        let result: f32 = Self::next_item(iter, lineno)?;
        Result::Ok(result)
    }

    fn next_usize<'a, I>(iter: &mut I, lineno: usize) -> Result<usize, Box<dyn std::error::Error>>
    where
        I: Iterator<Item = &'a str>,
    {
        let result: usize = Self::next_item(iter, lineno)?;
        Result::Ok(result)
    }

    fn next_coords(
        iter: &mut SplitWhitespace,
        lineno: usize,
    ) -> Result<(usize, usize, usize), Box<dyn std::error::Error>> {
        let mut result = iter.next().ok_or(Self::err_string(lineno))?.split("/");
        let x = Self::next_usize(&mut result, lineno)? - 1;
        let mut y = Self::next_usize(&mut result, lineno).unwrap_or(usize::MAX);
        if y != usize::MAX {
            y -= 1;
        }
        let z = Self::next_usize(&mut result, lineno)? - 1;
        Result::Ok((x, y, z))
    }

    pub fn load<P>(filename: P) -> Result<Mesh, Box<dyn std::error::Error>>
    where
        P: AsRef<Path>,
    {
        let lines = Self::read_lines(filename)?;
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut tex_coords = Vec::new();
        let mut normal_coords = Vec::new();

        for (lineno, line) in lines.enumerate() {
            match line {
                Ok(line_str) => {
                    let mut iter = line_str.split_whitespace();
                    let t = iter.next().unwrap_or("");
                    match t {
                        "v" => {
                            let x = Self::next_f32(&mut iter, lineno)?;
                            let y = Self::next_f32(&mut iter, lineno)?;
                            let z: f32 = Self::next_f32(&mut iter, lineno)?;
                            vertices.push(x);
                            vertices.push(y);
                            vertices.push(z);
                        }
                        "vt" => {
                            let u = Self::next_f32(&mut iter, lineno)?;
                            let v = Self::next_f32(&mut iter, lineno)?;
                            tex_coords.push(u);
                            tex_coords.push(v);
                        }
                        "vn" => {
                            let x = Self::next_f32(&mut iter, lineno)?;
                            let y = Self::next_f32(&mut iter, lineno)?;
                            let z = Self::next_f32(&mut iter, lineno)?;
                            normal_coords.push(x);
                            normal_coords.push(y);
                            normal_coords.push(z);
                        }
                        "f" => {
                            let (v1, vt1, vn1) = Self::next_coords(&mut iter, lineno)?;
                            let (v2, vt2, vn2) = Self::next_coords(&mut iter, lineno)?;
                            let (v3, vt3, vn3) = Self::next_coords(&mut iter, lineno)?;
                            indices.push(Index::new(v1, vt1, vn1));
                            indices.push(Index::new(v2, vt2, vn2));
                            indices.push(Index::new(v3, vt3, vn3));
                        }
                        _ => {}
                    }
                }
                err => {
                    err?;
                }
            }
        }
        Result::Ok(Mesh {
            vertices,
            indices,
            tex_coords,
            normal_coords,
        })
    }
}
