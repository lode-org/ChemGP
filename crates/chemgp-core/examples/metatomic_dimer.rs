mod shared {
    include!("rpc_dimer.rs");
}

fn main() {
    shared::main();
}
