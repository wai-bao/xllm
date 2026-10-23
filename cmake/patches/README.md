# Dependency patches

`etcd-cancellable-requests.patch` applies to the pinned `etcd_cpp_apiv3`
submodule. It makes lease cancellation interrupt an outstanding refresh before
joining its thread, cancels the stream before waiting for its final status, and
drains completion tags before destroying the queue. Stream creation also uses
the configured request timeout.

An authenticated `etcd::SyncClient` constructor overload accepts a final
`std::chrono::microseconds` argument. This sets the timeout before initial
authentication while preserving the existing constructor. Token renewal reads
the client's current timeout. `EtcdClient` uses the new overload.

`cmake/etcd_dependency.cmake` hashes the dependency sources and patch, then
prepares the patched source under the build directory. Both dependency
compilation and xLLM headers use that copy. The original submodule checkout and
gitlink are unchanged, and a patch conflict fails configuration.

The host-only `xservice_client_test` covers a withheld lease response, withheld
final stream status, initial authentication, and authentication renewal. These
tests exercise real gRPC transport without initializing an accelerator.
