use std::fmt::{Display, Formatter, Write};

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub struct DebugOpScopeRef(usize);
impl Display for DebugOpScopeRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
pub struct DebugOpRegistry {
    pub(crate) ops: Vec<DebugOp>,
}

impl DebugOpRegistry {
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn get_or_create(&mut self, op: DebugOp) -> DebugOpScopeRef {
        if let Some((index, _)) = self.ops.iter().enumerate().find(|(_, o)| o == &&op) {
            DebugOpScopeRef(index)
        } else {
            self.ops.push(op);
            DebugOpScopeRef(self.ops.len() - 1)
        }
    }
}

impl DebugOpRegistry {
    pub fn emit(&self, output: &mut String) {
        writeln!(output).unwrap();

        for (index, op) in self.ops.iter().enumerate() {
            write!(output, "!{} = ", index).unwrap();
            op.emit(output);
            writeln!(output).unwrap();
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum DebugOp {
    DIFile {
        filename: String,
        directory: String,
    },
    DICompileUnit {
        file: DebugOpScopeRef,
    },
    DISubprogram {
        name: String,
        scope: DebugOpScopeRef,
        file: DebugOpScopeRef,
        line: i32,
        unit: DebugOpScopeRef,
    },
    DILocation {
        line: i32,
        column: i32,
        scope: DebugOpScopeRef,
    },
    Raw(String),
}

impl DebugOp {
    pub fn emit(&self, output: &mut String) {
        match self {
            DebugOp::DIFile {
                filename,
                directory,
            } => {
                write!(
                    output,
                    "!DIFile(filename: \"{filename}\", directory: \"{directory}\")",
                )
                .unwrap();
            }
            DebugOp::DICompileUnit { file } => {
                write!(
                    output,
                    "distinct !DICompileUnit(language: DW_LANG_Rust, file: !{file}, emissionKind: FullDebug)"
                )
                .unwrap();
            }
            DebugOp::DISubprogram {
                name,
                scope,
                file,
                line,
                unit,
            } => {
                write!(output, r#"distinct !DISubprogram(name: "{name}", scope: !{scope}, file: !{file}, line: {line}, unit: !{unit})"#).unwrap();
            }
            DebugOp::DILocation {
                line,
                column,
                scope,
            } => {
                write!(
                    output,
                    r#"!DILocation(line: {line}, column: {column}, scope: !{scope})"#
                )
                .unwrap();
            }
            DebugOp::Raw(raw_str) => {
                write!(output, "{}", raw_str).unwrap();
            }
        }
    }
}
