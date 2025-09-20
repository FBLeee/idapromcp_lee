import os
import sys

if sys.version_info < (3, 11):
    raise RuntimeError("Python 3.11 or higher is required for the MCP plugin")

import json
import re
import struct
import threading
import http.server
from urllib.parse import urlparse
from typing import Any, Callable, get_type_hints, TypedDict, Optional, Annotated, TypeVar, Generic, NotRequired


class JSONRPCError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data


class RPCRegistry:
    def __init__(self):
        self.methods: dict[str, Callable] = {}
        self.unsafe: set[str] = set()

    def register(self, func: Callable) -> Callable:
        self.methods[func.__name__] = func
        return func

    def mark_unsafe(self, func: Callable) -> Callable:
        self.unsafe.add(func.__name__)
        return func

    def dispatch(self, method: str, params: Any) -> Any:
        if method not in self.methods:
            raise JSONRPCError(-32601, f"Method '{method}' not found")

        func = self.methods[method]
        hints = get_type_hints(func)

        # Remove return annotation if present
        hints.pop("return", None)

        if isinstance(params, list):
            if len(params) != len(hints):
                raise JSONRPCError(-32602, f"Invalid params: expected {len(hints)} arguments, got {len(params)}")

            # Validate and convert parameters
            converted_params = []
            for value, (param_name, expected_type) in zip(params, hints.items()):
                try:
                    if not isinstance(value, expected_type):
                        value = expected_type(value)
                    converted_params.append(value)
                except (ValueError, TypeError):
                    raise JSONRPCError(-32602,
                                       f"Invalid type for parameter '{param_name}': expected {expected_type.__name__}")

            return func(*converted_params)
        elif isinstance(params, dict):
            if set(params.keys()) != set(hints.keys()):
                raise JSONRPCError(-32602, f"Invalid params: expected {list(hints.keys())}")

            # Validate and convert parameters
            converted_params = {}
            for param_name, expected_type in hints.items():
                value = params.get(param_name)
                try:
                    if not isinstance(value, expected_type):
                        value = expected_type(value)
                    converted_params[param_name] = value
                except (ValueError, TypeError):
                    raise JSONRPCError(-32602,
                                       f"Invalid type for parameter '{param_name}': expected {expected_type.__name__}")

            return func(**converted_params)
        else:
            raise JSONRPCError(-32600, "Invalid Request: params must be array or object")


rpc_registry = RPCRegistry()


def jsonrpc(func: Callable) -> Callable:
    """Decorator to register a function as a JSON-RPC method"""
    global rpc_registry
    return rpc_registry.register(func)


def unsafe(func: Callable) -> Callable:
    """Decorator to register mark a function as unsafe"""
    return rpc_registry.mark_unsafe(func)


class JSONRPCRequestHandler(http.server.BaseHTTPRequestHandler):
    def send_jsonrpc_error(self, code: int, message: str, id: Any = None):
        response = {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            }
        }
        if id is not None:
            response["id"] = id
        response_body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response_body))
        self.end_headers()
        self.wfile.write(response_body)

    def do_POST(self):
        global rpc_registry

        parsed_path = urlparse(self.path)
        if parsed_path.path != "/mcp":
            self.send_jsonrpc_error(-32098, "Invalid endpoint", None)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_jsonrpc_error(-32700, "Parse error: missing request body", None)
            return

        request_body = self.rfile.read(content_length)
        try:
            request = json.loads(request_body)
        except json.JSONDecodeError:
            self.send_jsonrpc_error(-32700, "Parse error: invalid JSON", None)
            return

        # Prepare the response
        response = {
            "jsonrpc": "2.0"
        }
        if request.get("id") is not None:
            response["id"] = request.get("id")

        try:
            # Basic JSON-RPC validation
            if not isinstance(request, dict):
                raise JSONRPCError(-32600, "Invalid Request")
            if request.get("jsonrpc") != "2.0":
                raise JSONRPCError(-32600, "Invalid JSON-RPC version")
            if "method" not in request:
                raise JSONRPCError(-32600, "Method not specified")

            # Dispatch the method
            result = rpc_registry.dispatch(request["method"], request.get("params", []))
            response["result"] = result

        except JSONRPCError as e:
            response["error"] = {
                "code": e.code,
                "message": e.message
            }
            if e.data is not None:
                response["error"]["data"] = e.data
        except IDAError as e:
            response["error"] = {
                "code": -32000,
                "message": e.message,
            }
        except Exception as e:
            traceback.print_exc()
            response["error"] = {
                "code": -32603,
                "message": "Internal error (please report a bug)",
                "data": traceback.format_exc(),
            }

        try:
            response_body = json.dumps(response).encode("utf-8")
        except Exception as e:
            traceback.print_exc()
            response_body = json.dumps({
                "error": {
                    "code": -32603,
                    "message": "Internal error (please report a bug)",
                    "data": traceback.format_exc(),
                }
            }).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response_body))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format, *args):
        # Suppress logging
        pass


class MCPHTTPServer(http.server.HTTPServer):
    allow_reuse_address = False


# 最后对该Server启动和停止的封装
class Server:
    HOST = "localhost"
    PORT = 13337

    def __init__(self):
        self.server = None
        self.server_thread = None
        self.running = False

    def start(self):
        if self.running:
            print("[MCP] Server is already running")
            return

        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.running = True
        self.server_thread.start()

    def stop(self):
        if not self.running:
            return

        self.running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join()
            self.server = None
        print("[MCP] Server stopped")

    def _run_server(self):
        try:
            # Create server in the thread to handle binding
            self.server = MCPHTTPServer((Server.HOST, Server.PORT), JSONRPCRequestHandler)
            print(f"[MCP] Server started at http://{Server.HOST}:{Server.PORT}")
            self.server.serve_forever()
        except OSError as e:
            if e.errno == 98 or e.errno == 10048:  # Port already in use (Linux/Windows)
                print("[MCP] Error: Port 13337 is already in use")
            else:
                print(f"[MCP] Server error: {e}")
            self.running = False
        except Exception as e:
            print(f"[MCP] Server error: {e}")
        finally:
            self.running = False


# A module that helps with writing thread safe ida code.
# Based on:
# https://web.archive.org/web/20160305190440/http://www.williballenthin.com/blog/2015/09/04/idapython-synchronization-decorator/
import logging
import queue
import traceback
import functools

import ida_hexrays
import ida_kernwin
import ida_funcs
import ida_gdl
import ida_lines
import ida_idaapi
import idc
import idaapi
import idautils
import ida_nalt
import ida_bytes
import ida_typeinf
import ida_xref
import ida_entry
import idautils
import ida_idd
import ida_dbg
import ida_name
import ida_ida
import ida_frame


class IDAError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

    @property
    def message(self) -> str:
        return self.args[0]


class IDASyncError(Exception):
    pass


class DecompilerLicenseError(IDAError):
    pass


# Important note: Always make sure the return value from your function f is a
# copy of the data you have gotten from IDA, and not the original data.
#
# Example:
# --------
#
# Do this:
#
#   @idaread
#   def ts_Functions():
#       return list(idautils.Functions())
#
# Don't do this:
#
#   @idaread
#   def ts_Functions():
#       return idautils.Functions()
#

logger = logging.getLogger(__name__)


# Enum for safety modes. Higher means safer:
class IDASafety:
    ida_kernwin.MFF_READ
    SAFE_NONE = ida_kernwin.MFF_FAST
    SAFE_READ = ida_kernwin.MFF_READ
    SAFE_WRITE = ida_kernwin.MFF_WRITE


call_stack = queue.LifoQueue()


def sync_wrapper(ff, safety_mode: IDASafety):
    """
    Call a function ff with a specific IDA safety_mode.
    """
    # logger.debug('sync_wrapper: {}, {}'.format(ff.__name__, safety_mode))

    if safety_mode not in [IDASafety.SAFE_READ, IDASafety.SAFE_WRITE]:
        error_str = 'Invalid safety mode {} over function {}' \
            .format(safety_mode, ff.__name__)
        logger.error(error_str)
        raise IDASyncError(error_str)

    # No safety level is set up:
    res_container = queue.Queue()

    def runned():
        # logger.debug('Inside runned')

        # Make sure that we are not already inside a sync_wrapper:
        if not call_stack.empty():
            last_func_name = call_stack.get()
            error_str = ('Call stack is not empty while calling the '
                         'function {} from {}').format(ff.__name__, last_func_name)
            # logger.error(error_str)
            raise IDASyncError(error_str)

        call_stack.put((ff.__name__))
        try:
            res_container.put(ff())
        except Exception as x:
            res_container.put(x)
        finally:
            call_stack.get()
            # logger.debug('Finished runned')

    ret_val = idaapi.execute_sync(runned, safety_mode)
    res = res_container.get()
    if isinstance(res, Exception):
        raise res
    return res


def idawrite(f):
    """
    decorator for marking a function as modifying the IDB.
    schedules a request to be made in the main IDA loop to avoid IDB corruption.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        ff = functools.partial(f, *args, **kwargs)
        ff.__name__ = f.__name__
        return sync_wrapper(ff, idaapi.MFF_WRITE)

    return wrapper


def idaread(f):
    """
    decorator for marking a function as reading from the IDB.
    schedules a request to be made in the main IDA loop to avoid
      inconsistent results.
    MFF_READ constant via: http://www.openrce.org/forums/posts/1827
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        ff = functools.partial(f, *args, **kwargs)
        ff.__name__ = f.__name__
        return sync_wrapper(ff, idaapi.MFF_READ)

    return wrapper


def is_window_active():
    """Returns whether IDA is currently active"""
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        return False

    app = QApplication.instance()
    if app is None:
        return False

    for widget in app.topLevelWidgets():
        if widget.isActiveWindow():
            return True
    return False


class Metadata(TypedDict):
    path: str
    module: str
    base: str
    size: str
    md5: str
    sha256: str
    crc32: str
    filesize: str


def get_image_size() -> int:
    try:
        # https://www.hex-rays.com/products/ida/support/sdkdoc/structidainfo.html
        info = idaapi.get_inf_structure()
        omin_ea = info.omin_ea
        omax_ea = info.omax_ea
    except AttributeError:
        import ida_ida
        omin_ea = ida_ida.inf_get_omin_ea()
        omax_ea = ida_ida.inf_get_omax_ea()
    # Bad heuristic for image size (bad if the relocations are the last section)
    image_size = omax_ea - omin_ea
    # Try to extract it from the PE header
    header = idautils.peutils_t().header()
    if header and header[:4] == b"PE\0\0":
        image_size = struct.unpack("<I", header[0x50:0x54])[0]
    return image_size


@jsonrpc
@idaread
def get_metadata() -> Metadata:
    """Get metadata about the current IDB"""

    # Fat Mach-O binaries can return a None hash:
    # https://github.com/mrexodia/ida-pro-mcp/issues/26
    def hash(f):
        try:
            return f().hex()
        except:
            return None

    return Metadata(path=idaapi.get_input_file_path(),
                    module=idaapi.get_root_filename(),
                    base=hex(idaapi.get_imagebase()),
                    size=hex(get_image_size()),
                    md5=hash(ida_nalt.retrieve_input_file_md5),
                    sha256=hash(ida_nalt.retrieve_input_file_sha256),
                    crc32=hex(ida_nalt.retrieve_input_file_crc32()),
                    filesize=hex(ida_nalt.retrieve_input_file_size()))


def get_prototype(fn: ida_funcs.func_t) -> Optional[str]:
    try:
        prototype: ida_typeinf.tinfo_t = fn.get_prototype()
        if prototype is not None:
            return str(prototype)
        else:
            return None
    except AttributeError:
        try:
            return idc.get_type(fn.start_ea)
        except:
            tif = ida_typeinf.tinfo_t()
            if ida_nalt.get_tinfo(tif, fn.start_ea):
                return str(tif)
            return None
    except Exception as e:
        print(f"Error getting function prototype: {e}")
        return None


class Function(TypedDict):
    address: str
    name: str
    size: str


def parse_address(address: str) -> int:
    try:
        return int(address, 0)
    except ValueError:
        for ch in address:
            if ch not in "0123456789abcdefABCDEF":
                raise IDAError(f"Failed to parse address: {address}")
        raise IDAError(f"Failed to parse address (missing 0x prefix): {address}")


def get_function(address: int, *, raise_error=True) -> Function:
    fn = idaapi.get_func(address)
    if fn is None:
        if raise_error:
            raise IDAError(f"No function found at address {hex(address)}")
        return None

    try:
        name = fn.get_name()
    except AttributeError:
        name = ida_funcs.get_func_name(fn.start_ea)

    return Function(address=hex(address), name=name, size=hex(fn.end_ea - fn.start_ea))


DEMANGLED_TO_EA = {}


def create_demangled_to_ea_map():
    for ea in idautils.Functions():
        # Get the function name and demangle it
        # MNG_NODEFINIT inhibits everything except the main name
        # where default demangling adds the function signature
        # and decorators (if any)
        demangled = idaapi.demangle_name(
            idc.get_name(ea, 0), idaapi.MNG_NODEFINIT)
        if demangled:
            DEMANGLED_TO_EA[demangled] = ea


def get_type_by_name(type_name: str) -> ida_typeinf.tinfo_t:
    # 8-bit integers
    if type_name in ('int8', '__int8', 'int8_t', 'char', 'signed char'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT8)
    elif type_name in ('uint8', '__uint8', 'uint8_t', 'unsigned char', 'byte', 'BYTE'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT8)

    # 16-bit integers
    elif type_name in ('int16', '__int16', 'int16_t', 'short', 'short int', 'signed short', 'signed short int'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT16)
    elif type_name in ('uint16', '__uint16', 'uint16_t', 'unsigned short', 'unsigned short int', 'word', 'WORD'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT16)

    # 32-bit integers
    elif type_name in (
    'int32', '__int32', 'int32_t', 'int', 'signed int', 'long', 'long int', 'signed long', 'signed long int'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT32)
    elif type_name in (
    'uint32', '__uint32', 'uint32_t', 'unsigned int', 'unsigned long', 'unsigned long int', 'dword', 'DWORD'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT32)

    # 64-bit integers
    elif type_name in (
    'int64', '__int64', 'int64_t', 'long long', 'long long int', 'signed long long', 'signed long long int'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT64)
    elif type_name in (
    'uint64', '__uint64', 'uint64_t', 'unsigned int64', 'unsigned long long', 'unsigned long long int', 'qword',
    'QWORD'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT64)

    # 128-bit integers
    elif type_name in ('int128', '__int128', 'int128_t', '__int128_t'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT128)
    elif type_name in ('uint128', '__uint128', 'uint128_t', '__uint128_t', 'unsigned int128'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT128)

    # Floating point types
    elif type_name in ('float',):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_FLOAT)
    elif type_name in ('double',):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_DOUBLE)
    elif type_name in ('long double', 'ldouble'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_LDOUBLE)

    # Boolean type
    elif type_name in ('bool', '_Bool', 'boolean'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_BOOL)

    # Void type
    elif type_name in ('void',):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_VOID)

    # If not a standard type, try to get a named type
    tif = ida_typeinf.tinfo_t()
    if tif.get_named_type(None, type_name, ida_typeinf.BTF_STRUCT):
        return tif

    if tif.get_named_type(None, type_name, ida_typeinf.BTF_TYPEDEF):
        return tif

    if tif.get_named_type(None, type_name, ida_typeinf.BTF_ENUM):
        return tif

    if tif.get_named_type(None, type_name, ida_typeinf.BTF_UNION):
        return tif

    if tif := ida_typeinf.tinfo_t(type_name):
        return tif

    raise IDAError(f"Unable to retrieve {type_name} type info object")


@jsonrpc
@idaread
def get_function_by_name(
        name: Annotated[str, "Name of the function to get"]
) -> Function:
    """Get a function by its name"""
    function_address = idaapi.get_name_ea(idaapi.BADADDR, name)
    if function_address == idaapi.BADADDR:
        # If map has not been created yet, create it
        if len(DEMANGLED_TO_EA) == 0:
            create_demangled_to_ea_map()
        # Try to find the function in the map, else raise an error
        if name in DEMANGLED_TO_EA:
            function_address = DEMANGLED_TO_EA[name]
        else:
            raise IDAError(f"No function found with name {name}")
    return get_function(function_address)


@jsonrpc
@idaread
def get_function_by_address(
        address: Annotated[str, "Address of the function to get"],
) -> Function:
    """Get a function by its address"""
    return get_function(parse_address(address))


@jsonrpc
@idaread
def get_current_address() -> str:
    """Get the address currently selected by the user"""
    return hex(idaapi.get_screen_ea())


@jsonrpc
@idaread
def get_current_function() -> Optional[Function]:
    """Get the function currently selected by the user"""
    return get_function(idaapi.get_screen_ea())


class ConvertedNumber(TypedDict):
    decimal: str
    hexadecimal: str
    bytes: str
    ascii: Optional[str]
    binary: str


@jsonrpc
def convert_number(
        text: Annotated[str, "Textual representation of the number to convert"],
        size: Annotated[Optional[int], "Size of the variable in bytes"],
) -> ConvertedNumber:
    """Convert a number (decimal, hexadecimal) to different representations"""
    try:
        value = int(text, 0)
    except ValueError:
        raise IDAError(f"Invalid number: {text}")

    # Estimate the size of the number
    if not size:
        size = 0
        n = abs(value)
        while n:
            size += 1
            n >>= 1
        size += 7
        size //= 8

    # Convert the number to bytes
    try:
        bytes = value.to_bytes(size, "little", signed=True)
    except OverflowError:
        raise IDAError(f"Number {text} is too big for {size} bytes")

    # Convert the bytes to ASCII
    ascii = ""
    for byte in bytes.rstrip(b"\x00"):
        if byte >= 32 and byte <= 126:
            ascii += chr(byte)
        else:
            ascii = None
            break

    return ConvertedNumber(
        decimal=str(value),
        hexadecimal=hex(value),
        bytes=bytes.hex(" "),
        ascii=ascii,
        binary=bin(value),
    )


T = TypeVar("T")


class Page(TypedDict, Generic[T]):
    data: list[T]
    next_offset: Optional[int]


def paginate(data: list[T], offset: int, count: int) -> Page[T]:
    if count == 0:
        count = len(data)
    next_offset = offset + count
    if next_offset >= len(data):
        next_offset = None
    return {
        "data": data[offset:offset + count],
        "next_offset": next_offset,
    }


def pattern_filter(data: list[T], pattern: str, key: str) -> list[T]:
    if not pattern:
        return data

    # TODO: implement /regex/ matching

    def matches(item: T) -> bool:
        return pattern.lower() in item[key].lower()

    return list(filter(matches, data))


@jsonrpc
@idaread
def list_functions(
        offset: Annotated[int, "Offset to start listing from (start at 0)"],
        count: Annotated[int, "Number of functions to list (100 is a good default, 0 means remainder)"],
) -> Page[Function]:
    """List all functions in the database (paginated)"""
    functions = [get_function(address) for address in idautils.Functions()]
    return paginate(functions, offset, count)


class Global(TypedDict):
    address: str
    name: str


@jsonrpc
@idaread
def list_globals_filter(
        offset: Annotated[int, "Offset to start listing from (start at 0)"],
        count: Annotated[int, "Number of globals to list (100 is a good default, 0 means remainder)"],
        filter: Annotated[
            str, "Filter to apply to the list (required parameter, empty string for no filter). Case-insensitive contains or /regex/ syntax"],
) -> Page[Global]:
    """List matching globals in the database (paginated, filtered)"""
    globals = []
    for addr, name in idautils.Names():
        # Skip functions
        if not idaapi.get_func(addr):
            globals += [Global(address=hex(addr), name=name)]

    globals = pattern_filter(globals, filter, "name")
    return paginate(globals, offset, count)


@jsonrpc
def list_globals(
        offset: Annotated[int, "Offset to start listing from (start at 0)"],
        count: Annotated[int, "Number of globals to list (100 is a good default, 0 means remainder)"],
) -> Page[Global]:
    """List all globals in the database (paginated)"""
    return list_globals_filter(offset, count, "")


class Import(TypedDict):
    address: str
    imported_name: str
    module: str


@jsonrpc
@idaread
def list_imports(
        offset: Annotated[int, "Offset to start listing from (start at 0)"],
        count: Annotated[int, "Number of imports to list (100 is a good default, 0 means remainder)"],
) -> Page[Import]:
    """ List all imported symbols with their name and module (paginated) """
    nimps = ida_nalt.get_import_module_qty()

    rv = []
    for i in range(nimps):
        module_name = ida_nalt.get_import_module_name(i)
        if not module_name:
            module_name = "<unnamed>"

        def imp_cb(ea, symbol_name, ordinal, acc):
            if not symbol_name:
                symbol_name = f"#{ordinal}"

            acc += [Import(address=hex(ea), imported_name=symbol_name, module=module_name)]

            return True

        imp_cb_w_context = lambda ea, symbol_name, ordinal: imp_cb(ea, symbol_name, ordinal, rv)
        ida_nalt.enum_import_names(i, imp_cb_w_context)

    return paginate(rv, offset, count)


class String(TypedDict):
    address: str
    length: int
    string: str


@jsonrpc
@idaread
def list_strings_filter(
        offset: Annotated[int, "Offset to start listing from (start at 0)"],
        count: Annotated[int, "Number of strings to list (100 is a good default, 0 means remainder)"],
        filter: Annotated[
            str, "Filter to apply to the list (required parameter, empty string for no filter). Case-insensitive contains or /regex/ syntax"],
) -> Page[String]:
    """List matching strings in the database (paginated, filtered)"""
    strings = []
    for item in idautils.Strings():
        try:
            string = str(item)
            if string:
                strings += [
                    String(address=hex(item.ea), length=item.length, string=string),
                ]
        except:
            continue
    strings = pattern_filter(strings, filter, "string")
    return paginate(strings, offset, count)


@jsonrpc
def list_strings(
        offset: Annotated[int, "Offset to start listing from (start at 0)"],
        count: Annotated[int, "Number of strings to list (100 is a good default, 0 means remainder)"],
) -> Page[String]:
    """List all strings in the database (paginated)"""
    return list_strings_filter(offset, count, "")


@jsonrpc
@idaread
def list_local_types():
    """List all Local types in the database"""
    error = ida_hexrays.hexrays_failure_t()
    locals = []
    idati = ida_typeinf.get_idati()
    type_count = ida_typeinf.get_ordinal_limit(idati)
    for ordinal in range(1, type_count):
        try:
            tif = ida_typeinf.tinfo_t()
            if tif.get_numbered_type(idati, ordinal):
                type_name = tif.get_type_name()
                if not type_name:
                    type_name = f"<Anonymous Type #{ordinal}>"
                locals.append(f"\nType #{ordinal}: {type_name}")
                if tif.is_udt():
                    c_decl_flags = (
                                ida_typeinf.PRTYPE_MULTI | ida_typeinf.PRTYPE_TYPE | ida_typeinf.PRTYPE_SEMI | ida_typeinf.PRTYPE_DEF | ida_typeinf.PRTYPE_METHODS | ida_typeinf.PRTYPE_OFFSETS)
                    c_decl_output = tif._print(None, c_decl_flags)
                    if c_decl_output:
                        locals.append(f"  C declaration:\n{c_decl_output}")
                else:
                    simple_decl = tif._print(None,
                                             ida_typeinf.PRTYPE_1LINE | ida_typeinf.PRTYPE_TYPE | ida_typeinf.PRTYPE_SEMI)
                    if simple_decl:
                        locals.append(f"  Simple declaration:\n{simple_decl}")
            else:
                message = f"\nType #{ordinal}: Failed to retrieve information."
                if error.str:
                    message += f": {error.str}"
                if error.errea != idaapi.BADADDR:
                    message += f"from (address: {hex(error.errea)})"
                raise IDAError(message)
        except:
            continue
    return locals


def decompile_checked(address: int) -> ida_hexrays.cfunc_t:
    if not ida_hexrays.init_hexrays_plugin():
        raise IDAError("Hex-Rays decompiler is not available")
    error = ida_hexrays.hexrays_failure_t()
    cfunc: ida_hexrays.cfunc_t = ida_hexrays.decompile_func(address, error, ida_hexrays.DECOMP_WARNINGS)
    if not cfunc:
        if error.code == ida_hexrays.MERR_LICENSE:
            raise DecompilerLicenseError(
                "Decompiler licence is not available. Use `disassemble_function` to get the assembly code instead.")

        message = f"Decompilation failed at {hex(address)}"
        if error.str:
            message += f": {error.str}"
        if error.errea != idaapi.BADADDR:
            message += f" (address: {hex(error.errea)})"
        raise IDAError(message)
    return cfunc


@jsonrpc
@idaread
def decompile_function(
        address: Annotated[str, "Address of the function to decompile"],
) -> str:
    """Decompile a function at the given address"""
    address = parse_address(address)
    cfunc = decompile_checked(address)
    if is_window_active():
        ida_hexrays.open_pseudocode(address, ida_hexrays.OPF_REUSE)
    sv = cfunc.get_pseudocode()
    pseudocode = ""
    for i, sl in enumerate(sv):
        sl: ida_kernwin.simpleline_t
        item = ida_hexrays.ctree_item_t()
        addr = None if i > 0 else cfunc.entry_ea
        if cfunc.get_line_item(sl.line, 0, False, None, item, None):
            ds = item.dstr().split(": ")
            if len(ds) == 2:
                try:
                    addr = int(ds[0], 16)
                except ValueError:
                    pass
        line = ida_lines.tag_remove(sl.line)
        if len(pseudocode) > 0:
            pseudocode += "\n"
        if not addr:
            pseudocode += f"/* line: {i} */ {line}"
        else:
            pseudocode += f"/* line: {i}, address: {hex(addr)} */ {line}"

    return pseudocode


class DisassemblyLine(TypedDict):
    segment: NotRequired[str]
    address: str
    label: NotRequired[str]
    instruction: str
    comments: NotRequired[list[str]]


class Argument(TypedDict):
    name: str
    type: str


class DisassemblyFunction(TypedDict):
    name: str
    start_ea: str
    return_type: NotRequired[str]
    arguments: NotRequired[list[Argument]]
    stack_frame: list[dict]
    lines: list[DisassemblyLine]


@jsonrpc
@idaread
def disassemble_function(
        start_address: Annotated[str, "Address of the function to disassemble"],
) -> DisassemblyFunction:
    """Get assembly code for a function"""
    start = parse_address(start_address)
    func: ida_funcs.func_t = idaapi.get_func(start)
    if not func:
        raise IDAError(f"No function found containing address {start_address}")
    if is_window_active():
        ida_kernwin.jumpto(start)

    lines = []
    for address in ida_funcs.func_item_iterator_t(func):
        seg = idaapi.getseg(address)
        segment = idaapi.get_segm_name(seg) if seg else None

        label = idc.get_name(address, 0)
        if label and label == func.name and address == func.start_ea:
            label = None
        if label == "":
            label = None

        comments = []
        if comment := idaapi.get_cmt(address, False):
            comments += [comment]
        if comment := idaapi.get_cmt(address, True):
            comments += [comment]

        raw_instruction = idaapi.generate_disasm_line(address, 0)
        tls = ida_kernwin.tagged_line_sections_t()
        ida_kernwin.parse_tagged_line_sections(tls, raw_instruction)
        insn_section = tls.first(ida_lines.COLOR_INSN)

        operands = []
        for op_tag in range(ida_lines.COLOR_OPND1, ida_lines.COLOR_OPND8 + 1):
            op_n = tls.first(op_tag)
            if not op_n:
                break

            op: str = op_n.substr(raw_instruction)
            op_str = ida_lines.tag_remove(op)

            # Do a lot of work to add address comments for symbols
            for idx in range(len(op) - 2):
                if op[idx] != idaapi.COLOR_ON:
                    continue

                idx += 1
                if ord(op[idx]) != idaapi.COLOR_ADDR:
                    continue

                idx += 1
                addr_string = op[idx:idx + idaapi.COLOR_ADDR_SIZE]
                idx += idaapi.COLOR_ADDR_SIZE

                addr = int(addr_string, 16)

                # Find the next color and slice until there
                symbol = op[idx:op.find(idaapi.COLOR_OFF, idx)]

                if symbol == '':
                    # We couldn't figure out the symbol, so use the whole op_str
                    symbol = op_str

                comments += [f"{symbol}={addr:#x}"]

                # print its value if its type is available
                try:
                    value = get_global_variable_value_internal(addr)
                except:
                    continue

                comments += [f"*{symbol}={value}"]

            operands += [op_str]

        mnem = ida_lines.tag_remove(insn_section.substr(raw_instruction))
        instruction = f"{mnem} {', '.join(operands)}"

        line = DisassemblyLine(
            address=f"{address:#x}",
            instruction=instruction,
        )

        if len(comments) > 0:
            line.update(comments=comments)

        if segment:
            line.update(segment=segment)

        if label:
            line.update(label=label)

        lines += [line]

    prototype = func.get_prototype()
    arguments: list[Argument] = [Argument(name=arg.name, type=f"{arg.type}") for arg in
                                 prototype.iter_func()] if prototype else None

    disassembly_function = DisassemblyFunction(
        name=func.name,
        start_ea=f"{func.start_ea:#x}",
        stack_frame=get_stack_frame_variables_internal(func.start_ea),
        lines=lines
    )

    if prototype:
        disassembly_function.update(return_type=f"{prototype.get_rettype()}")

    if arguments:
        disassembly_function.update(arguments=arguments)

    return disassembly_function


class Xref(TypedDict):
    address: str
    type: str
    function: Optional[Function]


@jsonrpc
@idaread
def get_xrefs_to(
        address: Annotated[str, "Address to get cross references to"],
) -> list[Xref]:
    """Get all cross references to the given address"""
    xrefs = []
    xref: ida_xref.xrefblk_t
    for xref in idautils.XrefsTo(parse_address(address)):
        xrefs += [
            Xref(address=hex(xref.frm),
                 type="code" if xref.iscode else "data",
                 function=get_function(xref.frm, raise_error=False))
        ]
    return xrefs


@jsonrpc
@idaread
def get_xrefs_to_field(
        struct_name: Annotated[str, "Name of the struct (type) containing the field"],
        field_name: Annotated[str, "Name of the field (member) to get xrefs to"],
) -> list[Xref]:
    """Get all cross references to a named struct field (member)"""

    # Get the type library
    til = ida_typeinf.get_idati()
    if not til:
        raise IDAError("Failed to retrieve type library.")

    # Get the structure type info
    tif = ida_typeinf.tinfo_t()
    if not tif.get_named_type(til, struct_name, ida_typeinf.BTF_STRUCT, True, False):
        print(f"Structure '{struct_name}' not found.")
        return []

    # Get The field index
    idx = ida_typeinf.get_udm_by_fullname(None, struct_name + '.' + field_name)
    if idx == -1:
        print(f"Field '{field_name}' not found in structure '{struct_name}'.")
        return []

    # Get the type identifier
    tid = tif.get_udm_tid(idx)
    if tid == ida_idaapi.BADADDR:
        raise IDAError(f"Unable to get tid for structure '{struct_name}' and field '{field_name}'.")

    # Get xrefs to the tid
    xrefs = []
    xref: ida_xref.xrefblk_t
    for xref in idautils.XrefsTo(tid):
        xrefs += [
            Xref(address=hex(xref.frm),
                 type="code" if xref.iscode else "data",
                 function=get_function(xref.frm, raise_error=False))
        ]
    return xrefs


@jsonrpc
@idaread
def get_entry_points() -> list[Function]:
    """Get all entry points in the database"""
    result = []
    for i in range(ida_entry.get_entry_qty()):
        ordinal = ida_entry.get_entry_ordinal(i)
        address = ida_entry.get_entry(ordinal)
        func = get_function(address, raise_error=False)
        if func is not None:
            result.append(func)
    return result


@jsonrpc
@idawrite
def set_comment(
        address: Annotated[str, "Address in the function to set the comment for"],
        comment: Annotated[str, "Comment text"],
):
    """Set a comment for a given address in the function disassembly and pseudocode"""
    address = parse_address(address)

    if not idaapi.set_cmt(address, comment, False):
        raise IDAError(f"Failed to set disassembly comment at {hex(address)}")

    if not ida_hexrays.init_hexrays_plugin():
        return

    # Reference: https://cyber.wtf/2019/03/22/using-ida-python-to-analyze-trickbot/
    # Check if the address corresponds to a line
    try:
        cfunc = decompile_checked(address)
    except DecompilerLicenseError:
        # We failed to decompile the function due to a decompiler license error
        return

    # Special case for function entry comments
    if address == cfunc.entry_ea:
        idc.set_func_cmt(address, comment, True)
        cfunc.refresh_func_ctext()
        return

    eamap = cfunc.get_eamap()
    if address not in eamap:
        print(f"Failed to set decompiler comment at {hex(address)}")
        return
    nearest_ea = eamap[address][0].ea

    # Remove existing orphan comments
    if cfunc.has_orphan_cmts():
        cfunc.del_orphan_cmts()
        cfunc.save_user_cmts()

    # Set the comment by trying all possible item types
    tl = idaapi.treeloc_t()
    tl.ea = nearest_ea
    for itp in range(idaapi.ITP_SEMI, idaapi.ITP_COLON):
        tl.itp = itp
        cfunc.set_user_cmt(tl, comment)
        cfunc.save_user_cmts()
        cfunc.refresh_func_ctext()
        if not cfunc.has_orphan_cmts():
            return
        cfunc.del_orphan_cmts()
        cfunc.save_user_cmts()
    print(f"Failed to set decompiler comment at {hex(address)}")


def refresh_decompiler_widget():
    widget = ida_kernwin.get_current_widget()
    if widget is not None:
        vu = ida_hexrays.get_widget_vdui(widget)
        if vu is not None:
            vu.refresh_ctext()


def refresh_decompiler_ctext(function_address: int):
    error = ida_hexrays.hexrays_failure_t()
    cfunc: ida_hexrays.cfunc_t = ida_hexrays.decompile_func(function_address, error, ida_hexrays.DECOMP_WARNINGS)
    if cfunc:
        cfunc.refresh_func_ctext()


@jsonrpc
@idawrite
def rename_local_variable(
        function_address: Annotated[str, "Address of the function containing the variable"],
        old_name: Annotated[str, "Current name of the variable"],
        new_name: Annotated[str, "New name for the variable (empty for a default name)"],
):
    """Rename a local variable in a function"""
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    if not ida_hexrays.rename_lvar(func.start_ea, old_name, new_name):
        raise IDAError(f"Failed to rename local variable {old_name} in function {hex(func.start_ea)}")
    refresh_decompiler_ctext(func.start_ea)


@jsonrpc
@idawrite
def rename_global_variable(
        old_name: Annotated[str, "Current name of the global variable"],
        new_name: Annotated[str, "New name for the global variable (empty for a default name)"],
):
    """Rename a global variable"""
    ea = idaapi.get_name_ea(idaapi.BADADDR, old_name)
    if not idaapi.set_name(ea, new_name):
        raise IDAError(f"Failed to rename global variable {old_name} to {new_name}")
    refresh_decompiler_ctext(ea)


@jsonrpc
@idawrite
def set_global_variable_type(
        variable_name: Annotated[str, "Name of the global variable"],
        new_type: Annotated[str, "New type for the variable"],
):
    """Set a global variable's type"""
    ea = idaapi.get_name_ea(idaapi.BADADDR, variable_name)
    tif = get_type_by_name(new_type)
    if not tif:
        raise IDAError(f"Parsed declaration is not a variable type")
    if not ida_typeinf.apply_tinfo(ea, tif, ida_typeinf.PT_SIL):
        raise IDAError(f"Failed to apply type")


@jsonrpc
@idaread
def get_global_variable_value_by_name(variable_name: Annotated[str, "Name of the global variable"]) -> str:
    """
    Read a global variable's value (if known at compile-time)

    Prefer this function over the `data_read_*` functions.
    """
    ea = idaapi.get_name_ea(idaapi.BADADDR, variable_name)
    if ea == idaapi.BADADDR:
        raise IDAError(f"Global variable {variable_name} not found")

    return get_global_variable_value_internal(ea)


@jsonrpc
@idaread
def get_global_variable_value_at_address(ea: Annotated[str, "Address of the global variable"]) -> str:
    """
    Read a global variable's value by its address (if known at compile-time)

    Prefer this function over the `data_read_*` functions.
    """
    ea = parse_address(ea)
    return get_global_variable_value_internal(ea)


def get_global_variable_value_internal(ea: int) -> str:
    # Get the type information for the variable
    tif = ida_typeinf.tinfo_t()
    if not ida_nalt.get_tinfo(tif, ea):
        # No type info, maybe we can figure out its size by its name
        if not ida_bytes.has_any_name(ea):
            raise IDAError(f"Failed to get type information for variable at {ea:#x}")

        size = ida_bytes.get_item_size(ea)
        if size == 0:
            raise IDAError(f"Failed to get type information for variable at {ea:#x}")
    else:
        # Determine the size of the variable
        size = tif.get_size()

    # Read the value based on the size
    if size == 0 and tif.is_array() and tif.get_array_element().is_decl_char():
        return_string = idaapi.get_strlit_contents(ea, -1, 0).decode("utf-8").strip()
        return f"\"{return_string}\""
    elif size == 1:
        return hex(ida_bytes.get_byte(ea))
    elif size == 2:
        return hex(ida_bytes.get_word(ea))
    elif size == 4:
        return hex(ida_bytes.get_dword(ea))
    elif size == 8:
        return hex(ida_bytes.get_qword(ea))
    else:
        # For other sizes, return the raw bytes
        return ' '.join(hex(x) for x in ida_bytes.get_bytes(ea, size))


@jsonrpc
@idawrite
def rename_function(
        function_address: Annotated[str, "Address of the function to rename"],
        new_name: Annotated[str, "New name for the function (empty for a default name)"],
):
    """Rename a function"""
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    if not idaapi.set_name(func.start_ea, new_name):
        raise IDAError(f"Failed to rename function {hex(func.start_ea)} to {new_name}")
    refresh_decompiler_ctext(func.start_ea)


@jsonrpc
@idawrite
def set_function_prototype(
        function_address: Annotated[str, "Address of the function"],
        prototype: Annotated[str, "New function prototype"],
):
    """Set a function's prototype"""
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    try:
        tif = ida_typeinf.tinfo_t(prototype, None, ida_typeinf.PT_SIL)
        if not tif.is_func():
            raise IDAError(f"Parsed declaration is not a function type")
        if not ida_typeinf.apply_tinfo(func.start_ea, tif, ida_typeinf.PT_SIL):
            raise IDAError(f"Failed to apply type")
        refresh_decompiler_ctext(func.start_ea)
    except Exception as e:
        raise IDAError(f"Failed to parse prototype string: {prototype}")


class my_modifier_t(ida_hexrays.user_lvar_modifier_t):
    def __init__(self, var_name: str, new_type: ida_typeinf.tinfo_t):
        ida_hexrays.user_lvar_modifier_t.__init__(self)
        self.var_name = var_name
        self.new_type = new_type

    def modify_lvars(self, lvars):
        for lvar_saved in lvars.lvvec:
            lvar_saved: ida_hexrays.lvar_saved_info_t
            if lvar_saved.name == self.var_name:
                lvar_saved.type = self.new_type
                return True
        return False


# NOTE: This is extremely hacky, but necessary to get errors out of IDA
def parse_decls_ctypes(decls: str, hti_flags: int) -> tuple[int, str]:
    if sys.platform == "win32":
        import ctypes

        assert isinstance(decls, str), "decls must be a string"
        assert isinstance(hti_flags, int), "hti_flags must be an int"
        c_decls = decls.encode("utf-8")
        c_til = None
        ida_dll = ctypes.CDLL("ida")
        ida_dll.parse_decls.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        ida_dll.parse_decls.restype = ctypes.c_int

        messages = []

        @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)
        def magic_printer(fmt: bytes, arg1: bytes):
            if fmt.count(b"%") == 1 and b"%s" in fmt:
                formatted = fmt.replace(b"%s", arg1)
                messages.append(formatted.decode("utf-8"))
                return len(formatted) + 1
            else:
                messages.append(f"unsupported magic_printer fmt: {repr(fmt)}")
                return 0

        errors = ida_dll.parse_decls(c_til, c_decls, magic_printer, hti_flags)
    else:
        # NOTE: The approach above could also work on other platforms, but it's
        # not been tested and there are differences in the vararg ABIs.
        errors = ida_typeinf.parse_decls(None, decls, False, hti_flags)
        messages = []
    return errors, messages


@jsonrpc
@idawrite
def declare_c_type(
        c_declaration: Annotated[
            str, "C declaration of the type. Examples include: typedef int foo_t; struct bar { int a; bool b; };"],
):
    """Create or update a local type from a C declaration"""
    # PT_SIL: Suppress warning dialogs (although it seems unnecessary here)
    # PT_EMPTY: Allow empty types (also unnecessary?)
    # PT_TYP: Print back status messages with struct tags
    flags = ida_typeinf.PT_SIL | ida_typeinf.PT_EMPTY | ida_typeinf.PT_TYP
    errors, messages = parse_decls_ctypes(c_declaration, flags)

    pretty_messages = "\n".join(messages)
    if errors > 0:
        raise IDAError(f"Failed to parse type:\n{c_declaration}\n\nErrors:\n{pretty_messages}")
    return f"success\n\nInfo:\n{pretty_messages}"


@jsonrpc
@idawrite
def set_local_variable_type(
        function_address: Annotated[str, "Address of the decompiled function containing the variable"],
        variable_name: Annotated[str, "Name of the variable"],
        new_type: Annotated[str, "New type for the variable"],
):
    """Set a local variable's type"""
    try:
        # Some versions of IDA don't support this constructor
        new_tif = ida_typeinf.tinfo_t(new_type, None, ida_typeinf.PT_SIL)
    except Exception:
        try:
            new_tif = ida_typeinf.tinfo_t()
            # parse_decl requires semicolon for the type
            ida_typeinf.parse_decl(new_tif, None, new_type + ";", ida_typeinf.PT_SIL)
        except Exception:
            raise IDAError(f"Failed to parse type: {new_type}")
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    if not ida_hexrays.rename_lvar(func.start_ea, variable_name, variable_name):
        raise IDAError(f"Failed to find local variable: {variable_name}")
    modifier = my_modifier_t(variable_name, new_tif)
    if not ida_hexrays.modify_user_lvars(func.start_ea, modifier):
        raise IDAError(f"Failed to modify local variable: {variable_name}")
    refresh_decompiler_ctext(func.start_ea)


class StackFrameVariable(TypedDict):
    name: str
    offset: str
    size: str
    type: str


@jsonrpc
@idaread
def get_stack_frame_variables(
        function_address: Annotated[str, "Address of the disassembled function to retrieve the stack frame variables"]
) -> list[StackFrameVariable]:
    """ Retrieve the stack frame variables for a given function """
    return get_stack_frame_variables_internal(parse_address(function_address))


def get_stack_frame_variables_internal(function_address: int) -> list[dict]:
    func = idaapi.get_func(function_address)
    if not func:
        raise IDAError(f"No function found at address {function_address}")

    members = []
    tif = ida_typeinf.tinfo_t()
    if not tif.get_type_by_tid(func.frame) or not tif.is_udt():
        return []

    udt = ida_typeinf.udt_type_data_t()
    tif.get_udt_details(udt)
    for udm in udt:
        if not udm.is_gap():
            name = udm.name
            offset = udm.offset // 8
            size = udm.size // 8
            type = str(udm.type)

            members += [StackFrameVariable(name=name,
                                           offset=hex(offset),
                                           size=hex(size),
                                           type=type)
                        ]

    return members


class StructureMember(TypedDict):
    name: str
    offset: str
    size: str
    type: str


class StructureDefinition(TypedDict):
    name: str
    size: str
    members: list[StructureMember]


@jsonrpc
@idaread
def get_defined_structures() -> list[StructureDefinition]:
    """ Returns a list of all defined structures """

    rv = []
    limit = ida_typeinf.get_ordinal_limit()
    for ordinal in range(1, limit):
        tif = ida_typeinf.tinfo_t()
        tif.get_numbered_type(None, ordinal)
        if tif.is_udt():
            udt = ida_typeinf.udt_type_data_t()
            members = []
            if tif.get_udt_details(udt):
                members = [
                    StructureMember(name=x.name,
                                    offset=hex(x.offset // 8),
                                    size=hex(x.size // 8),
                                    type=str(x.type))
                    for _, x in enumerate(udt)
                ]

            rv += [StructureDefinition(name=tif.get_type_name(),
                                       size=hex(tif.get_size()),
                                       members=members)]

    return rv


@jsonrpc
@idawrite
def rename_stack_frame_variable(
        function_address: Annotated[str, "Address of the disassembled function to set the stack frame variables"],
        old_name: Annotated[str, "Current name of the variable"],
        new_name: Annotated[str, "New name for the variable (empty for a default name)"]
):
    """ Change the name of a stack variable for an IDA function """
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")

    frame_tif = ida_typeinf.tinfo_t()
    if not ida_frame.get_func_frame(frame_tif, func):
        raise IDAError("No frame returned.")

    idx, udm = frame_tif.get_udm(old_name)
    if not udm:
        raise IDAError(f"{old_name} not found.")

    tid = frame_tif.get_udm_tid(idx)
    if ida_frame.is_special_frame_member(tid):
        raise IDAError(f"{old_name} is a special frame member. Will not change the name.")

    udm = ida_typeinf.udm_t()
    frame_tif.get_udm_by_tid(udm, tid)
    offset = udm.offset // 8
    if ida_frame.is_funcarg_off(func, offset):
        raise IDAError(f"{old_name} is an argument member. Will not change the name.")

    sval = ida_frame.soff_to_fpoff(func, offset)
    if not ida_frame.define_stkvar(func, new_name, sval, udm.type):
        raise IDAError("failed to rename stack frame variable")


@jsonrpc
@idawrite
def create_stack_frame_variable(
        function_address: Annotated[str, "Address of the disassembled function to set the stack frame variables"],
        offset: Annotated[str, "Offset of the stack frame variable"],
        variable_name: Annotated[str, "Name of the stack variable"],
        type_name: Annotated[str, "Type of the stack variable"]
):
    """ For a given function, create a stack variable at an offset and with a specific type """

    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")

    offset = parse_address(offset)

    frame_tif = ida_typeinf.tinfo_t()
    if not ida_frame.get_func_frame(frame_tif, func):
        raise IDAError("No frame returned.")

    tif = get_type_by_name(type_name)
    if not ida_frame.define_stkvar(func, variable_name, offset, tif):
        raise IDAError("failed to define stack frame variable")


@jsonrpc
@idawrite
def set_stack_frame_variable_type(
        function_address: Annotated[str, "Address of the disassembled function to set the stack frame variables"],
        variable_name: Annotated[str, "Name of the stack variable"],
        type_name: Annotated[str, "Type of the stack variable"]
):
    """ For a given disassembled function, set the type of a stack variable """

    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")

    frame_tif = ida_typeinf.tinfo_t()
    if not ida_frame.get_func_frame(frame_tif, func):
        raise IDAError("No frame returned.")

    idx, udm = frame_tif.get_udm(variable_name)
    if not udm:
        raise IDAError(f"{variable_name} not found.")

    tid = frame_tif.get_udm_tid(idx)
    udm = ida_typeinf.udm_t()
    frame_tif.get_udm_by_tid(udm, tid)
    offset = udm.offset // 8

    tif = get_type_by_name(type_name)
    if not ida_frame.set_frame_member_type(func, offset, tif):
        raise IDAError("failed to set stack frame variable type")


@jsonrpc
@idawrite
def delete_stack_frame_variable(
        function_address: Annotated[str, "Address of the function to set the stack frame variables"],
        variable_name: Annotated[str, "Name of the stack variable"]
):
    """ Delete the named stack variable for a given function """

    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")

    frame_tif = ida_typeinf.tinfo_t()
    if not ida_frame.get_func_frame(frame_tif, func):
        raise IDAError("No frame returned.")

    idx, udm = frame_tif.get_udm(variable_name)
    if not udm:
        raise IDAError(f"{variable_name} not found.")

    tid = frame_tif.get_udm_tid(idx)
    if ida_frame.is_special_frame_member(tid):
        raise IDAError(f"{variable_name} is a special frame member. Will not delete.")

    udm = ida_typeinf.udm_t()
    frame_tif.get_udm_by_tid(udm, tid)
    offset = udm.offset // 8
    size = udm.size // 8
    if ida_frame.is_funcarg_off(func, offset):
        raise IDAError(f"{variable_name} is an argument member. Will not delete.")

    if not ida_frame.delete_frame_members(func, offset, offset + size):
        raise IDAError("failed to delete stack frame variable")


@jsonrpc
@idaread
def read_memory_bytes(
        memory_address: Annotated[str, "Address of the memory value to be read"],
        size: Annotated[int, "size of memory to read"]
) -> str:
    """
    Read bytes at a given address.

    Only use this function if `get_global_variable_at` and `get_global_variable_by_name`
    both failed.
    """
    return ' '.join(f'{x:#02x}' for x in ida_bytes.get_bytes(parse_address(memory_address), size))


@jsonrpc
@idaread
def data_read_byte(
        address: Annotated[str, "Address to get 1 byte value from"],
) -> int:
    """
    Read the 1 byte value at the specified address.

    Only use this function if `get_global_variable_at` failed.
    """
    ea = parse_address(address)
    return ida_bytes.get_wide_byte(ea)


@jsonrpc
@idaread
def data_read_word(
        address: Annotated[str, "Address to get 2 bytes value from"],
) -> int:
    """
    Read the 2 byte value at the specified address as a WORD.

    Only use this function if `get_global_variable_at` failed.
    """
    ea = parse_address(address)
    return ida_bytes.get_wide_word(ea)


@jsonrpc
@idaread
def data_read_dword(
        address: Annotated[str, "Address to get 4 bytes value from"],
) -> int:
    """
    Read the 4 byte value at the specified address as a DWORD.

    Only use this function if `get_global_variable_at` failed.
    """
    ea = parse_address(address)
    return ida_bytes.get_wide_dword(ea)


@jsonrpc
@idaread
def data_read_qword(
        address: Annotated[str, "Address to get 8 bytes value from"]
) -> int:
    """
    Read the 8 byte value at the specified address as a QWORD.

    Only use this function if `get_global_variable_at` failed.
    """
    ea = parse_address(address)
    return ida_bytes.get_qword(ea)


@jsonrpc
@idaread
def data_read_string(
        address: Annotated[str, "Address to get string from"]
) -> str:
    """
    Read the string at the specified address.

    Only use this function if `get_global_variable_at` failed.
    """
    try:
        return idaapi.get_strlit_contents(parse_address(address), -1, 0).decode("utf-8")
    except Exception as e:
        return "Error:" + str(e)


@jsonrpc
@idaread
@unsafe
def dbg_get_registers() -> list[dict[str, str]]:
    """Get all registers and their values. This function is only available when debugging."""
    result = []
    dbg = ida_idd.get_dbg()
    # TODO: raise an exception when not debugging?
    for thread_index in range(ida_dbg.get_thread_qty()):
        tid = ida_dbg.getn_thread(thread_index)
        regs = []
        regvals = ida_dbg.get_reg_vals(tid)
        for reg_index, rv in enumerate(regvals):
            reg_info = dbg.regs(reg_index)
            reg_value = rv.pyval(reg_info.dtype)
            if isinstance(reg_value, int):
                reg_value = hex(reg_value)
            if isinstance(reg_value, bytes):
                reg_value = reg_value.hex(" ")
            regs.append({
                "name": reg_info.name,
                "value": reg_value,
            })
        result.append({
            "thread_id": tid,
            "registers": regs,
        })
    return result


@jsonrpc
@idaread
@unsafe
def dbg_get_call_stack() -> list[dict[str, str]]:
    """Get the current call stack."""
    callstack = []
    try:
        tid = ida_dbg.get_current_thread()
        trace = ida_idd.call_stack_t()

        if not ida_dbg.collect_stack_trace(tid, trace):
            return []
        for frame in trace:
            frame_info = {
                "address": hex(frame.callea),
            }
            try:
                module_info = ida_idd.modinfo_t()
                if ida_dbg.get_module_info(frame.callea, module_info):
                    frame_info["module"] = os.path.basename(module_info.name)
                else:
                    frame_info["module"] = "<unknown>"

                name = (
                        ida_name.get_nice_colored_name(
                            frame.callea,
                            ida_name.GNCN_NOCOLOR
                            | ida_name.GNCN_NOLABEL
                            | ida_name.GNCN_NOSEG
                            | ida_name.GNCN_PREFDBG,
                        )
                        or "<unnamed>"
                )
                frame_info["symbol"] = name

            except Exception as e:
                frame_info["module"] = "<error>"
                frame_info["symbol"] = str(e)

            callstack.append(frame_info)

    except Exception as e:
        pass
    return callstack


def list_breakpoints():
    ea = ida_ida.inf_get_min_ea()
    end_ea = ida_ida.inf_get_max_ea()
    breakpoints = []
    while ea <= end_ea:
        bpt = ida_dbg.bpt_t()
        if ida_dbg.get_bpt(ea, bpt):
            breakpoints.append(
                {
                    "ea": hex(bpt.ea),
                    "type": bpt.type,
                    "enabled": bpt.flags & ida_dbg.BPT_ENABLED,
                    "condition": bpt.condition if bpt.condition else None,
                }
            )
        ea = ida_bytes.next_head(ea, end_ea)
    return breakpoints


@jsonrpc
@idaread
@unsafe
def dbg_list_breakpoints():
    """List all breakpoints in the program."""
    return list_breakpoints()


@jsonrpc
@idaread
@unsafe
def dbg_start_process() -> str:
    """Start the debugger"""
    if idaapi.start_process("", "", ""):
        return "Debugger started"
    return "Failed to start debugger"


@jsonrpc
@idaread
@unsafe
def dbg_exit_process() -> str:
    """Exit the debugger"""
    if idaapi.exit_process():
        return "Debugger exited"
    return "Failed to exit debugger"


@jsonrpc
@idaread
@unsafe
def dbg_continue_process() -> str:
    """Continue the debugger"""
    if idaapi.continue_process():
        return "Debugger continued"
    return "Failed to continue debugger"


@jsonrpc
@idaread
@unsafe
def dbg_run_to(
        address: Annotated[str, "Run the debugger to the specified address"],
) -> str:
    """Run the debugger to the specified address"""
    ea = parse_address(address)
    if idaapi.run_to(ea):
        return f"Debugger run to {hex(ea)}"
    return f"Failed to run to address {hex(ea)}"


@jsonrpc
@idaread
@unsafe
def dbg_set_breakpoint(
        address: Annotated[str, "Set a breakpoint at the specified address"],
) -> str:
    """Set a breakpoint at the specified address"""
    ea = parse_address(address)
    if idaapi.add_bpt(ea, 0, idaapi.BPT_SOFT):
        return f"Breakpoint set at {hex(ea)}"
    breakpoints = list_breakpoints()
    for bpt in breakpoints:
        if bpt["ea"] == hex(ea):
            return f"Breakpoint already exists at {hex(ea)}"
    return f"Failed to set breakpoint at address {hex(ea)}"


@jsonrpc
@idaread
@unsafe
def dbg_delete_breakpoint(
        address: Annotated[str, "del a breakpoint at the specified address"],
) -> str:
    """del a breakpoint at the specified address"""
    ea = parse_address(address)
    if idaapi.del_bpt(ea):
        return f"Breakpoint deleted at {hex(ea)}"
    return f"Failed to delete breakpoint at address {hex(ea)}"


@jsonrpc
@idaread
@unsafe
def dbg_enable_breakpoint(
        address: Annotated[str, "Enable or disable a breakpoint at the specified address"],
        enable: Annotated[bool, "Enable or disable a breakpoint"],
) -> str:
    """Enable or disable a breakpoint at the specified address"""
    ea = parse_address(address)
    if idaapi.enable_bpt(ea, enable):
        return f"Breakpoint {'enabled' if enable else 'disabled'} at {hex(ea)}"
    return f"Failed to {'' if enable else 'disable '}breakpoint at address {hex(ea)}"


@jsonrpc
@idaread
def get_ida_version() -> str:
    """Get the current IDA Pro version."""
    return idaapi.get_kernel_version()


# @jsonrpc
# @idaread
# def find_potential_key_functions() -> list[Function]:
#     """
#     查找可能包含硬编码密钥或执行类似密钥操作的函数。
#     """
#     potential_key_functions_set = set()

#     common_crypto_functions = [
#         "AES_encrypt", "AES_decrypt", "RC4", "MD5_Update", "SHA1_Update", "SHA256_Update",
#         "CryptEncrypt", "CryptDecrypt", "EVP_EncryptInit_ex", "EVP_DecryptInit_ex",
#         "RSA_public_encrypt", "RSA_private_decrypt", "DES_encrypt", "Blowfish_encrypt",
#         "EVP_BytesToKey", "EVP_CipherInit_ex", "SSL_CTX_use_certificate_file",
#         "RSA_set_key", "DSA_set_key", "EC_KEY_set_private_key"
#     ]
#     common_crypto_strings = [
#         "AES", "RSA", "MD5", "SHA", "key", "password", "secret", "cipher", "decrypt", "encrypt",
#         "EVP_CIPHER_CTX", "EVP_MD_CTX", "SSL_CTX", "BIO", "X509", "PKCS", "PEM", "DER",
#         "aes-128-cbc", "aes-256-cbc", "rc4", "des-cbc", "sha256", "md5",
#         "private key", "public key", "certificate", "random", "seed"
#     ]

#     # 模式1：硬编码字符串后跟带算术运算的循环，并检查特定长度
#     pattern_1 = r"qmemcpy\(v\d+, \"(.+?)\", (\d+)\);\s*.*?do\s*.*?v\d+\[n0x\d+\] = \((\d+ \* \(v\d+\[n0x\d+\] - \d+\) % \d+ \+ \d+\) % \d+);"

#     # 模式2：调用常见加密/哈希函数或存在加密相关字符串
#     crypto_func_pattern = r"\b(" + "|".join(re.escape(f) for f in common_crypto_functions) + r")\(.*?\);"
#     crypto_string_pattern = r"\b(" + "|".join(re.escape(s) for s in common_crypto_strings) + r")\b"
#     pattern_2_general = f"({crypto_func_pattern}|{crypto_string_pattern})"

#     # 模式3：访问常见的OpenSSL结构体成员（例如，ctx->key, rsa->n）
#     openssl_struct_members = [
#         r"\bctx->key\b", r"\bctx->iv\b", r"\bctx->cipher\b",
#         r"\brsa->n\b", r"\brsa->e\b", r"\brsa->d\b",
#         r"\baes_key->rd_key\b", r"\baes_key->rounds\b"
#     ]
#     pattern_3_struct_access = r"(" + "|".join(openssl_struct_members) + r")"

#     # 模式4：汇编级别特征
#     assembly_patterns = [
#         r"xor\s+r[a-z0-9]+,\s+r[a-z0-9]+",
#         r"rol\s+", r"ror\s+", r"shl\s+", r"shr\s+",
#         r"imul\s+",
#         r"mov\s+r[a-z0-9]+,\s+0x[0-9a-fA-F]{8,}",
#         r"call\s+ds:\[.*\]",
#         r"call\s+qword\s+ptr\s+\[.*\]"
#     ]
#     pattern_4_assembly = r"(" + "|".join(assembly_patterns) + r")"

#     # 阶段1：初步筛选 - 基于函数名、导入、字符串引用和特定模式的交叉引用
#     all_functions = {func_ea: get_function(func_ea) for func_ea in idautils.Functions()}

#     # 1.1 基于函数名筛选
#     for func_ea, func_info in all_functions.items():
#         if any(keyword.lower() in func_info["name"].lower() for keyword in common_crypto_functions + common_crypto_strings):
#             potential_key_functions_set.add(func_ea)

#     # 1.2 基于导入函数筛选 (通用加密函数和内存复制函数)
#     imports_page = list_imports(offset=0, count=0) # 获取所有导入
#     mem_copy_functions = ["qmemcpy", "memcpy", "RtlCopyMemory", "CopyMemory"] # 常见的内存复制函数

#     for imp in imports_page["data"]:
#         if any(keyword.lower() in imp["imported_name"].lower() for keyword in common_crypto_functions + mem_copy_functions):
#             xrefs = get_xrefs_to(imp["address"])
#             for xref in xrefs:
#                 if xref["function"]:
#                     potential_key_functions_set.add(parse_address(xref["function"]["address"]))

#     # 1.3 基于字符串筛选 (通用加密字符串)
#     strings_page = list_strings(offset=0, count=0) # 获取所有字符串
#     for s in strings_page["data"]:
#         if any(keyword.lower() in s["string"].lower() for keyword in common_crypto_strings):
#             xrefs = get_xrefs_to(s["address"])
#             for xref in xrefs:
#                 if xref["function"]:
#                     potential_key_functions_set.add(parse_address(xref["function"]["address"]))

#     # 阶段2：对初步筛选出的函数进行详细分析
#     final_potential_key_functions = []
#     for func_ea in potential_key_functions_set:
#         try:
#             is_potential_key_func = False
#             pseudocode = None
#             assembly_code = None

#             # 尝试反编译函数
#             try:
#                 pseudocode = decompile_function(hex(func_ea))
#             except DecompilerLicenseError:
#                 pass
#             except IDAError as e:
#                 print(f"Warning: Failed to decompile function {hex(func_ea)}: {e}")
#                 pass

#             # 尝试反汇编函数
#             try:
#                 disassembly = disassemble_function(hex(func_ea))
#                 assembly_code = "\n".join([line["instruction"] for line in disassembly["lines"]])
#             except IDAError as e:
#                 print(f"Warning: Failed to disassemble function {hex(func_ea)}: {e}")
#                 pass

#             if pseudocode:
#                 # 检查模式1：硬编码字符串后跟带算术运算的循环
#                 match_1 = re.search(pattern_1, pseudocode, re.DOTALL)
#                 if match_1:
#                     try:
#                         key_length = int(match_1.group(2))
#                         if key_length in [8, 16, 32]: # 常见的密钥长度
#                             is_potential_key_func = True
#                     except ValueError:
#                         pass

#                 # 检查模式2和模式3
#                 if not is_potential_key_func:
#                     if re.search(pattern_2_general, pseudocode, re.IGNORECASE | re.DOTALL):
#                         is_potential_key_func = True
#                     elif re.search(pattern_3_struct_access, pseudocode, re.IGNORECASE | re.DOTALL):
#                         is_potential_key_func = True

#             if assembly_code and not is_potential_key_func:
#                 # 检查模式4
#                 if re.search(pattern_4_assembly, assembly_code, re.IGNORECASE | re.DOTALL):
#                     is_potential_key_func = True

#             if is_potential_key_func:
#                 func_info = get_function(func_ea)
#                 final_potential_key_functions.append(func_info)
#                 set_comment(hex(func_ea), "MCP插件识别：潜在密钥/加密函数。")

#         except Exception as e:
#             print(f"Unexpected error during detailed analysis of function {hex(func_ea)}: {e}")
#             continue

#     return final_potential_key_functions


# ----------------------------------------------------------------------
# 正则模式
# ----------------------------------------------------------------------
_STACK_VAR_RE = re.compile(r'\[(?:e|r)bp\s*\+\s*var_([0-9A-Fa-f]+)\]')
_REG_PLUS_OFF_RE = re.compile(r'\[(?P<reg>[er](?:ax|bx|cx|dx|si|di|bp|sp)|r\d+)\s*\+\s*(?P<off>0x[0-9A-Fa-f]+|\d+)\]')


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def _imm_u8(ea, opn=1) -> bool:
    if idc.get_operand_type(ea, opn) != idc.o_imm:
        return False
    val = idc.get_operand_value(ea, opn)
    return 0 <= val <= 0xFF


def _parse_stack_var_offset(op_text: str):
    m = _STACK_VAR_RE.search(op_text)
    if not m:
        return None
    try:
        return int(m.group(1), 16)
    except Exception:
        return None


def _parse_reg_plus_off(op_text: str):
    m = _REG_PLUS_OFF_RE.search(op_text)
    if not m:
        return (None, None)
    reg = m.group('reg').lower()
    off = m.group('off')
    try:
        base = 16 if off.lower().startswith('0x') else 10
        return (reg, int(off, base))
    except Exception:
        return (None, None)


# ----------------------------------------------------------------------
# 特征 1：连续 imm8 写入本地缓冲（快速扫描）
# ----------------------------------------------------------------------
def _track_runs_over_func(func_ea: int):
    best_cnt, best_start_ea, best_tag = 0, None, None
    cur_cnt, cur_start_ea, last_off, tag = 0, None, None, None
    lea_bases = {}

    for ea in idautils.FuncItems(func_ea):
        mnem = idc.print_insn_mnem(ea).lower()

        # 追踪 lea reg, [rbp+var_XX]
        if mnem == 'lea':
            dst = idc.print_operand(ea, 0).strip().lower()
            src = idc.print_operand(ea, 1).strip().lower()
            base_off = _parse_stack_var_offset(src)
            if base_off is not None:
                lea_bases[dst] = base_off
            continue

        if mnem != 'mov':
            if cur_cnt > best_cnt:
                best_cnt, best_start_ea, best_tag = cur_cnt, cur_start_ea, tag
            cur_cnt, cur_start_ea, last_off, tag = 0, None, None, None
            continue

        if idc.get_operand_type(ea, 0) not in (idc.o_displ, idc.o_phrase, idc.o_mem):
            if cur_cnt > best_cnt:
                best_cnt, best_start_ea, best_tag = cur_cnt, cur_start_ea, tag
            cur_cnt, cur_start_ea, last_off, tag = 0, None, None, None
            continue

        if not _imm_u8(ea, 1):
            if cur_cnt > best_cnt:
                best_cnt, best_start_ea, best_tag = cur_cnt, cur_start_ea, tag
            cur_cnt, cur_start_ea, last_off, tag = 0, None, None, None
            continue

        mem_txt = idc.print_operand(ea, 0).lower()
        off = _parse_stack_var_offset(mem_txt)
        cur_tag = None
        if off is not None:
            cur_tag = 'stack'
        else:
            reg, add = _parse_reg_plus_off(mem_txt)
            if reg and add is not None and reg in lea_bases:
                off = lea_bases[reg] + add
                cur_tag = f'{reg}-based'
            else:
                if cur_cnt > best_cnt:
                    best_cnt, best_start_ea, best_tag = cur_cnt, cur_start_ea, tag
                cur_cnt, cur_start_ea, last_off, tag = 0, None, None, None
                continue

        if last_off is None:
            cur_cnt, cur_start_ea, last_off, tag = 1, ea, off, cur_tag
        else:
            if abs(off - last_off) == 1 and cur_tag == tag:
                cur_cnt += 1
                last_off = off
            else:
                if cur_cnt > best_cnt:
                    best_cnt, best_start_ea, best_tag = cur_cnt, cur_start_ea, tag
                cur_cnt, cur_start_ea, last_off, tag = 1, ea, off, cur_tag

    if cur_cnt > best_cnt:
        best_cnt, best_start_ea, best_tag = cur_cnt, cur_start_ea, tag

    return best_cnt, best_start_ea, best_tag


def find_const_byte_runs_fast(min_run: int = 5):
    results = []
    for f_ea in idautils.Functions():
        run_len, start_ea, mode = _track_runs_over_func(f_ea)
        if run_len >= min_run:
            name = idaapi.get_func_name(f_ea) or f"sub_{f_ea:08X}"
            results.append({
                "ea": f_ea,
                "name": name,
                "type": "const_bytes",
                "detail": f"{run_len} bytes ({mode})"
            })
    return results


# ----------------------------------------------------------------------
# 特征 2：检测 call 与 非 call 的特定 API 调用
# ----------------------------------------------------------------------
API_KEYWORDS = [
    "qmemcpy", "memcpy", "memmove", "memset", "strcpy", "strncpy",
    "AES_encrypt", "AES_decrypt", "RC4", "MD5_Update", "SHA1_Update", "SHA256_Update",
    "CryptEncrypt", "CryptDecrypt", "EVP_EncryptInit_ex", "EVP_DecryptInit_ex",
    "RSA_public_encrypt", "RSA_private_decrypt", "DES_encrypt", "Blowfish_encrypt",
    "EVP_BytesToKey", "EVP_CipherInit_ex", "SSL_CTX_use_certificate_file",
    "RSA_set_key", "DSA_set_key", "EC_KEY_set_private_key"
]


def find_api_refs():
    results = []
    for f_ea in idautils.Functions():
        name = idaapi.get_func_name(f_ea) or f"sub_{f_ea:08X}"
        for ea in idautils.FuncItems(f_ea):
            op1 = idc.print_operand(ea, 0).lower()
            op2 = idc.print_operand(ea, 1).lower()
            mnem = idc.print_insn_mnem(ea).lower()
            text = op1 + " " + op2
            if any(api in text for api in API_KEYWORDS):
                results.append({
                    "ea": f_ea,
                    "name": name,
                    "type": "api_ref",
                    "detail": f"{mnem} {text}"
                })
                break
    return results


# ----------------------------------------------------------------------
# 特征 3：检测可疑长度字符串（8 / 16 / 32，可能是密钥）或者找和加解密相关的
# ----------------------------------------------------------------------
def find_suspicious_strings():
    results = []
    crypto_strings = [
        "AES", "RSA", "MD5", "SHA", "key", "password", "secret", "cipher", "decrypt", "encrypt",
        "EVP_CIPHER_CTX", "EVP_MD_CTX", "SSL_CTX", "BIO", "X509", "PKCS", "PEM", "DER",
        "aes-128-cbc", "aes-256-cbc", "rc4", "des-cbc", "sha256", "md5",
        "private key", "public key", "certificate", "random", "seed"
    ]

    TYPE_WHITELIST = []
    for s in idautils.Strings():
        text = str(s).strip()
        length = len(text)
        is_suspicious = length in (8, 16, 32) or any(keyword.lower() in text.lower() for keyword in crypto_strings)
        if is_suspicious and text.lower() not in (t.lower() for t in TYPE_WHITELIST):
            refs = list(idautils.DataRefsTo(s.ea))
            for ref in refs:
                f_ea = idc.get_func_attr(ref, idc.FUNCATTR_START)
                if f_ea != idc.BADADDR:
                    name = idaapi.get_func_name(f_ea) or f"sub_{f_ea:08X}"
                    results.append({
                        "ea": f_ea,
                        "name": name,
                        "type": "sus_str",
                        "detail": f"len={length} '{text}'"
                    })
    return results


# ----------------------------------------------------------------------
# 处理 find_potential_key_functions 的模式 1、2、3 和 4：
# ----------------------------------------------------------------------
def find_crypto_patterns(min_functions=100):
    results = []
    # 初步筛选：基于函数名、导入和字符串
    potential_functions = set()
    all_functions = {func_ea: get_function(func_ea) for func_ea in idautils.Functions()}

    # 函数名筛选
    crypto_keywords = [
        "AES", "RSA", "MD5", "SHA", "key", "password", "secret", "cipher", "decrypt", "encrypt",
        "EVP", "SSL", "BIO", "X509", "PKCS", "PEM", "DER", "rc4", "des"
    ]
    for func_ea, func_info in all_functions.items():
        if any(keyword.lower() in func_info["name"].lower() for keyword in crypto_keywords):
            potential_functions.add(func_ea)

    # 导入筛选
    imports_page = list_imports(offset=0, count=0)
    for imp in imports_page["data"]:
        if any(keyword.lower() in imp["imported_name"].lower() for keyword in
               crypto_keywords + ["qmemcpy", "memcpy", "RtlCopyMemory", "CopyMemory"]):
            xrefs = get_xrefs_to(imp["address"])
            for xref in xrefs:
                if xref["function"]:
                    potential_functions.add(parse_address(xref["function"]["address"]))

    # 字符串筛选
    strings_page = list_strings(offset=0, count=0)
    for s in strings_page["data"]:
        if any(keyword.lower() in s["string"].lower() for keyword in crypto_keywords):
            xrefs = get_xrefs_to(s["address"])
            for xref in xrefs:
                if xref["function"]:
                    potential_functions.add(parse_address(xref["function"]["address"]))

    # 限制分析函数数量，防止卡死
    potential_functions = list(potential_functions)[:min_functions]

    # 正则表达式模式
    pattern_1 = r"qmemcpy\(v\d+, \"(.+?)\", (\d+)\);\s*.*?do\s*.*?v\d+\[n0x\d+\] = \((\d+ \* \(v\d+\[n0x\d+\] - \d+\) % \d+ \+ \d+\) % \d+);"
    crypto_func_pattern = r"\b(" + "|".join(re.escape(f) for f in [
        "AES_encrypt", "AES_decrypt", "RC4", "MD5_Update", "SHA1_Update", "SHA256_Update",
        "CryptEncrypt", "CryptDecrypt", "EVP_EncryptInit_ex", "EVP_DecryptInit_ex",
        "RSA_public_encrypt", "RSA_private_decrypt", "DES_encrypt", "Blowfish_encrypt",
        "EVP_BytesToKey", "EVP_CipherInit_ex", "SSL_CTX_use_certificate_file",
        "RSA_set_key", "DSA_set_key", "EC_KEY_set_private_key"
    ]) + r")\(.*?\);"
    crypto_string_pattern = r"\b(" + "|".join(re.escape(s) for s in crypto_keywords) + r")\b"
    pattern_2_general = f"({crypto_func_pattern}|{crypto_string_pattern})"
    pattern_3_struct_access = r"(" + "|".join([
        r"\bctx->key\b", r"\bctx->iv\b", r"\bctx->cipher\b",
        r"\brsa->n\b", r"\brsa->e\b", r"\brsa->d\b",
        r"\baes_key->rd_key\b", r"\baes_key->rounds\b"
    ]) + r")"
    pattern_4_assembly = r"(" + "|".join([
        r"xor\s+r[a-z0-9]+,\s+r[a-z0-9]+",
        r"rol\s+", r"ror\s+", r"shl\s+", r"shr\s+",
        r"imul\s+",
        r"mov\s+r[a-z0-9]+,\s+0x[0-9a-fA-F]{8,}",
        r"call\s+ds:\[.*\]",
        r"call\s+qword\s+ptr\s+\[.*\]"
    ]) + r")"

    for func_ea in potential_functions:
        try:
            is_potential_key_func = False
            # 反编译
            pseudocode = None
            try:
                pseudocode = decompile_function(hex(func_ea))
                if re.search(pattern_1, pseudocode, re.DOTALL):
                    key_length = int(re.search(pattern_1, pseudocode, re.DOTALL).group(2))
                    if key_length in [8, 16, 32]:
                        is_potential_key_func = True
                elif re.search(pattern_2_general, pseudocode, re.IGNORECASE | re.DOTALL):
                    is_potential_key_func = True
                elif re.search(pattern_3_struct_access, pseudocode, re.IGNORECASE | re.DOTALL):
                    is_potential_key_func = True
            except (DecompilerLicenseError, IDAError):
                pass

            # 反汇编
            if not is_potential_key_func:
                try:
                    disassembly = disassemble_function(hex(func_ea))
                    assembly_code = "\n".join([line["instruction"] for line in disassembly["lines"]])
                    if re.search(pattern_4_assembly, assembly_code, re.IGNORECASE | re.DOTALL):
                        is_potential_key_func = True
                except IDAError:
                    pass

            if is_potential_key_func:
                func_info = get_function(func_ea)
                results.append({
                    "ea": func_ea,
                    "name": func_info["name"],
                    "type": "crypto_pattern",
                    "detail": "Potential crypto/key function"
                })
                set_comment(hex(func_ea), "MCP插件识别：潜在密钥/加密函数。")
        except Exception as e:
            print(f"Error analyzing function {hex(func_ea)}: {e}")
            continue

    return results


# ----------------------------------------------------------------------
# 主扫描函数
# ----------------------------------------------------------------------
@jsonrpc
@idaread
def run_all_scans():
    """
    查找可能包含硬编码密钥或执行类似密钥操作的函数。
    """
    results = []
    results.extend(find_const_byte_runs_fast(5))
    results.extend(find_api_refs())
    results.extend(find_suspicious_strings())
    results.extend(find_crypto_patterns(min_functions=100))  # 限制为前 100 个候选函数

    print(f"[+] Found {len(results)} suspicious functions")
    for r in results:
        print(f"{r['ea']:08X} {r['name']:<30} {r['type']:<12} {r['detail']}")
    return results

# ----------------------------------------------------------------------
# 插件！！！！！MCP
# ----------------------------------------------------------------------
# ========== 动作 Handler（Server类定义在前面） ==========
class StartServerHandler(idaapi.action_handler_t):
    def __init__(self, server_instance):
        idaapi.action_handler_t.__init__(self)
        self.server = server_instance

    def activate(self, ctx):
        self.server.start()
        return 1

    def update(self, ctx):
        return idaapi.AST_ENABLE_ALWAYS


class StopServerHandler(idaapi.action_handler_t):
    def __init__(self, server_instance):
        idaapi.action_handler_t.__init__(self)
        self.server = server_instance

    def activate(self, ctx):
        self.server.stop()
        return 1

    def update(self, ctx):
        return idaapi.AST_ENABLE_ALWAYS


def register_mcp_menu():
    server = Server()
    idaapi.create_menu("Edit/MCP", "MCP")

    idaapi.register_action(idaapi.action_desc_t(
        "mcp:start_server",
        "Start MCP Server",
        StartServerHandler(server),
        "Ctrl-Alt-M",
        "Start the MCP server"
    ))
    idaapi.attach_action_to_menu("Edit/MCP/Start_MCP", "mcp:start_server", idaapi.SETMENU_APP)

    idaapi.register_action(idaapi.action_desc_t(
        "mcp:stop_server",
        "Stop MCP Server",
        StopServerHandler(server),
        "Ctrl-Alt-V",
        "Stop the MCP server"
    ))
    idaapi.attach_action_to_menu("Edit/MCP/Stop_MCP", "mcp:stop_server", idaapi.SETMENU_APP)


# 脚本加载时直接注册菜单
register_mcp_menu()



# ----------------------------------------------------------------------
# 插件！！！！！Frida代码生成器
# ----------------------------------------------------------------------
# ========== Frida代码生成逻辑类 ==========
class FridaCodeGenerator:
    def __init__(self, output_file="frida_hooks.js", frida_version="17.x"):
        self.output_file = os.path.abspath(output_file)
        self.frida_version = frida_version
        self.is_legacy_version = self._parse_version(frida_version) < (17, 0)
    
    def _parse_version(self, version_str):
        """解析版本字符串，返回版本元组"""
        try:
            if version_str.endswith('.x'):
                version_str = version_str[:-2]
            parts = version_str.split('.')
            major = int(parts[0]) if len(parts) > 0 else 16
            minor = int(parts[1]) if len(parts) > 1 else 0
            return (major, minor)
        except (ValueError, IndexError):
            # 默认为旧版本
            return (16, 0)

    def generate_hook_for_function(self, func_ea):
        """为指定函数生成Frida Hook代码"""
        try:
            func_info = get_function(func_ea)
            func_name = func_info["name"]
            func_addr = func_info["address"]
            
            # 获取函数原型
            func = idaapi.get_func(func_ea)
            prototype = None
            args_info = []
            ret_type = "void"
            
            if func:
                try:
                    prototype = get_prototype(func)
                    if prototype:
                        # 解析函数原型获取参数信息
                        args_info = self._parse_function_args(prototype)
                        ret_type = self._parse_return_type(prototype)
                except Exception as e:
                    print(f"Warning: Failed to get prototype for {func_name}: {e}")
            
            # 生成Frida Hook代码
            hook_code = self._generate_frida_hook_template(
                func_name, func_addr, args_info, ret_type
            )
            
            return hook_code
            
        except Exception as e:
            print(f"Error generating Frida code for function at {hex(func_ea)}: {e}")
            return None
    
    def _parse_function_args(self, prototype_str):
        """解析函数原型获取参数信息"""
        args = []
        try:
            # 简单的参数解析（可以根据需要改进）
            if '(' in prototype_str and ')' in prototype_str:
                args_part = prototype_str[prototype_str.find('(')+1:prototype_str.rfind(')')].strip()
                if args_part and args_part != 'void':
                    arg_list = [arg.strip() for arg in args_part.split(',')]
                    for i, arg in enumerate(arg_list):
                        parts = arg.split()
                        if len(parts) >= 2:
                            arg_type = ' '.join(parts[:-1])
                            arg_name = parts[-1].replace('*', '').replace('&', '')
                        else:
                            arg_type = arg
                            arg_name = f"arg{i}"
                        args.append({"type": arg_type, "name": arg_name})
        except Exception as e:
            print(f"Warning: Failed to parse function arguments: {e}")
        return args
    
    def _parse_return_type(self, prototype_str):
        """解析函数原型获取返回值类型"""
        try:
            if prototype_str:
                parts = prototype_str.split('(')
                if len(parts) > 0:
                    ret_part = parts[0].strip()
                    # 移除函数名
                    words = ret_part.split()
                    if len(words) > 1:
                        return ' '.join(words[:-1])
        except Exception:
            pass
        return "void"
    
    def _generate_frida_hook_template(self, func_name, func_addr, args_info, ret_type):
        """生成Frida Hook模板代码"""
        # 根据版本选择不同的模板
        if self.is_legacy_version:
            return self._generate_legacy_template(func_name, func_addr, args_info, ret_type)
        else:
            return self._generate_modern_template(func_name, func_addr, args_info, ret_type)
    
    def _generate_legacy_template(self, func_name, func_addr, args_info, ret_type):
        """生成Frida 17.x之前版本的Hook模板代码"""
        # 构建参数赋值部分（使用参数名而非 argX）
        param_assignments = []
        param_logs = []
        
        for i, arg in enumerate(args_info):
            param_name = arg['name'] if arg['name'] else f"arg{i}"
            param_assignments.append(f"            this.{param_name} = args[{i}];")
            
            arg_type = arg['type'].lower()
            if 'char*' in arg_type or 'string' in arg_type:
                param_logs.append(f"                `        {param_name}:\\n${{this.{param_name}}}\\n`,")
            elif any(t in arg_type for t in ['int', 'long', 'dword']):
                param_logs.append(f"                `        {param_name}:\\n${{this.{param_name}.toInt32()}}\\n`,")
            elif 'pointer' in arg_type or '*' in arg_type:
                # 指针类型不生成hexdump，按照用户示例格式
                param_logs.append(f"                `        {param_name}:\\n${{this.{param_name}.toInt32()}}\\n`,")
            else:
                param_logs.append(f"                `        {param_name}:\\n${{this.{param_name}}}\\n`,")
        
        # 构建返回值处理
        if ret_type != "void":
            ret_log = f"console.log(`[+] Leaving {func_name}, return: ${{retval}}`)"
        else:
            ret_log = f"console.log(`[+] Leaving {func_name}`)"
        
        # 生成参数输出部分
        assignments_str = "\n".join(param_assignments) if param_assignments else ""
        params_str = "\n".join(param_logs) if param_logs else ""
        
        # 获取函数地址的数值部分（去掉 0x 前缀）
        addr_value = func_addr[2:] if func_addr.startswith('0x') else func_addr
        
        # 生成模板
        template = f"""// Hook for function: {func_name} at {func_addr} (Frida <17.x)
function hook_{func_name}() {{
    var base_addr = Module.findBaseAddress("your_module.so"); // 请修改为实际模块名
    var {func_name}_addr = base_addr.add(0x{addr_value});  //注意，该地址就是函数的地址，THUMB要加0x1
    Interceptor.attach({func_name}_addr, {{
        onEnter: function(args) {{
{assignments_str}
            console.log(`\\n------------------{func_name}-------------\\n`,
                `[+] Entering {func_name}\\n`,
{params_str}
                `setValue called from:\\n` + Thread.backtrace(this.context, Backtracer.ACCURATE).map(DebugSymbol.fromAddress).join('\\n') + '\\r\\n'
            );
        }},
        onLeave: function(retval) {{
            {ret_log};
        }}
    }})
}}

"""
        
        return template
    
    def _generate_modern_template(self, func_name, func_addr, args_info, ret_type):
        """生成Frida 17.x及以后版本的Hook模板代码"""
        # 构建参数赋值部分（使用参数名而非 argX）
        param_assignments = []
        param_logs = []
        
        for i, arg in enumerate(args_info):
            param_name = arg['name'] if arg['name'] else f"arg{i}"
            param_assignments.append(f"            this.{param_name} = args[{i}];")
            
            arg_type = arg['type'].lower()
            if 'char*' in arg_type or 'string' in arg_type:
                param_logs.append(f"                `        {param_name}:\\n${{this.{param_name}}}\\n`,")
            elif any(t in arg_type for t in ['int', 'long', 'dword']):
                param_logs.append(f"                `        {param_name}:\\n${{this.{param_name}.toInt32()}}\\n`,")
            elif 'pointer' in arg_type or '*' in arg_type:
                # 指针类型不生成hexdump，按照用户示例格式
                param_logs.append(f"                `        {param_name}:\\n${{this.{param_name}.toInt32()}}\\n`,")
            else:
                param_logs.append(f"                `        {param_name}:\\n${{this.{param_name}}}\\n`,")
        
        # 构建返回值处理
        if ret_type != "void":
            ret_log = f"console.log(`[+] Leaving {func_name}, return: ${{retval}}`)"
        else:
            ret_log = f"console.log(`[+] Leaving {func_name}`)"
        
        # 生成参数输出部分
        assignments_str = "\n".join(param_assignments) if param_assignments else ""
        params_str = "\n".join(param_logs) if param_logs else ""
        
        # 获取函数地址的数值部分（去掉 0x 前缀）
        addr_value = func_addr[2:] if func_addr.startswith('0x') else func_addr
        
        # 生成模板
        template = f"""// Hook for function: {func_name} at {func_addr} (Frida >=17.x)
function hook_{func_name}() {{
    const base_addr = Process.getModuleByName("your_module.so").base; // 请修改为实际模块名
    const {func_name}_addr = base_addr.add(0x{addr_value});  //注意，该地址就是函数的地址，THUMB要加0x1
    Interceptor.attach({func_name}_addr, {{
        onEnter(args) {{
{assignments_str}
            console.log(`\\n------------------{func_name}-------------\\n`,
                `[+] Entering {func_name}\\n`,
{params_str}
                `setValue called from:\\n` + Thread.backtrace(this.context, Backtracer.ACCURATE).map(DebugSymbol.fromAddress).join('\\n') + '\\r\\n'
            );
        }},
        onLeave(retval) {{
            {ret_log};
        }}
    }})
}}

"""
        
        return template
    
    def generate_for_current_function(self, frida_version=None):
        """为当前选中的函数生成Frida代码"""
        # 如果是通过插件菜单调用，使用按钮弹窗询问版本
        if frida_version is None:
            try:
                import ida_kernwin
                # 使用按钮弹窗询问版本
                choice = ida_kernwin.ask_buttons(
                    ">=17", "<17", "Cancel",
                    ida_kernwin.ASKBTN_BTN1,
                    "Choose Frida version:"
                )
                
                if choice == ida_kernwin.ASKBTN_CANCEL:
                    print("[INFO] Operation cancelled")
                    return False
                elif choice == ida_kernwin.ASKBTN_BTN1:
                    frida_version = "17.x"
                else:  # ASKBTN_BTN2
                    frida_version = "16.x"
            except:
                frida_version = "17.x"  # 如果弹窗失败，使用默认版本
            
        self.frida_version = frida_version
        self.is_legacy_version = self._parse_version(frida_version) < (17, 0)
            
        current_ea = idaapi.get_screen_ea()
        func = idaapi.get_func(current_ea)
        
        if not func:
            print("[ERROR] No function found at current address")
            return False
        
        # 检查文件是否存在，如果存在则问是否覆盖
        import os
        if os.path.exists(self.output_file):
            try:
                import ida_kernwin
                overwrite = ida_kernwin.ask_buttons(
                    "Overwrite", "Cancel", "",
                    ida_kernwin.ASKBTN_BTN1,
                    f"File '{self.output_file}' already exists.\n\nDo you want to overwrite it?"
                )
                
                if overwrite != ida_kernwin.ASKBTN_BTN1:
                    print("[INFO] Operation cancelled - file not overwritten")
                    return False
            except:
                # 如果弹窗失败，默认不覆盖
                print("[WARNING] File exists, operation cancelled")
                return False
        
        hook_code = self.generate_hook_for_function(func.start_ea)
        if hook_code:
            with open(self.output_file, "w", encoding="utf-8") as f:  # 使用 "w" 模式覆盖写入
                f.write(hook_code)
                # 添加主函数和调用
                f.write(f"\nfunction main() {{\n")
                f.write(f"    hook_{idaapi.get_func_name(func.start_ea)}();\n")
                f.write(f"}}\n\n")
                f.write(f"setImmediate(main);\n")
            
            version_info = "<17.x" if self.is_legacy_version else ">=17.x"
            print(f"[SUCCESS] Frida hook code (v{version_info}) generated for {idaapi.get_func_name(func.start_ea)} and saved to {self.output_file}")
            return True
        return False
    


@jsonrpc
@idaread
def generate_frida_hooks_batch(
    function_addresses: Annotated[list[str], "List of function addresses to generate Frida hooks for"],
    frida_version: Annotated[str, "Frida version (e.g., '16.5', '17.0', '17.x'). Defaults to '17.x'"] = "17.x"
) -> list[dict]:
    """Generate Frida hook code for multiple functions"""
    results = []
    generator = FridaCodeGenerator(frida_version=frida_version)
    
    for addr_str in function_addresses:
        try:
            func_ea = parse_address(addr_str)
            hook_code = generator.generate_hook_for_function(func_ea)
            
            if hook_code:
                func_info = get_function(func_ea)
                results.append({
                    "address": addr_str,
                    "name": func_info["name"],
                    "hook_code": hook_code,
                    "frida_version": frida_version,
                    "success": True
                })
            else:
                results.append({
                    "address": addr_str,
                    "name": "unknown",
                    "hook_code": None,
                    "frida_version": frida_version,
                    "success": False,
                    "error": "Failed to generate hook code"
                })
        except Exception as e:
            results.append({
                "address": addr_str, 
                "name": "unknown",
                "hook_code": None,
                "frida_version": frida_version,
                "success": False,
                "error": str(e)
            })
    
    return results


@jsonrpc
@idaread
def get_frida_version_info() -> dict:
    """Get information about supported Frida versions and their differences"""
    return {
        "supported_versions": ["16.x", "17.x"],
        "default_version": "17.x",
        "version_differences": {
            "16.x": {
                "description": "Legacy Frida versions (before 17.0)",
                "features": [
                    "Traditional var declarations",
                    "function() syntax",
                    "String concatenation with +",
                    "Memory.readUtf8String() error handling",
                    "Classical JavaScript patterns"
                ],
                "limitations": [
                    "No modern ES6+ features",
                    "Less efficient memory operations",
                    "Basic error handling"
                ]
            },
            "17.x": {
                "description": "Modern Frida versions (17.0 and later)",
                "features": [
                    "ES6+ syntax (const/let, template literals)",
                    "Arrow functions support",
                    "Enhanced Memory API with ArrayBuffer.wrap()",
                    "Improved error handling",
                    "Modern JavaScript features",
                    "Better performance optimizations"
                ],
                "improvements": [
                    "More efficient memory access",
                    "Better debugging capabilities",
                    "Cleaner code generation",
                    "Enhanced type safety"
                ]
            }
        },
        "migration_notes": [
            "Frida 17.x introduces modern JavaScript features",
            "Legacy code should still work but modern syntax is recommended",
            "ArrayBuffer.wrap() provides more efficient memory access",
            "Template literals improve string formatting readability"
        ]
    }


# ----------------------------------------------------------------------
# 插件！！！！！反编译输出C代码
# ----------------------------------------------------------------------
# ========== 异步反编译逻辑类 ==========
class AsyncDecompileServer:
    def __init__(self, output_file="decompiled_functions.c"):
        self.output_file = os.path.abspath(output_file)

    def start(self):
        # 在主线程异步执行
        ida_kernwin.execute_sync(self._worker, ida_kernwin.MFF_NOWAIT)

    def _worker(self):
        import os, time
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        if not ida_hexrays.init_hexrays_plugin():
            print("[ERROR] Hex-Rays decompiler not available.")
            return

        func_list = list(idautils.Functions())
        total_functions = len(func_list)
        function_count = 0
        skipped_functions = 0
        start_time = time.time()

        with open(self.output_file, "w", encoding="utf-8") as f:
            for func_ea in func_list:
                function_count += 1
                func_start_time = time.time()
                func_name = idaapi.get_func_name(func_ea)
                func_addr = hex(func_ea)

                f.write(f"Function: {func_name} at {func_addr}\n==============\n")

                try:
                    cfunc = ida_hexrays.decompile(func_ea)
                    if cfunc:
                        pseudocode = "\n".join(
                            ida_lines.tag_remove(sl.line) for sl in cfunc.get_pseudocode()
                        )
                        f.write(pseudocode + "\n")
                    else:
                        f.write("Error: Failed to decompile function.\n")
                        skipped_functions += 1
                except Exception as e:
                    f.write(f"Error: {str(e)}\n")
                    skipped_functions += 1

                f.write("==============\n\n")
                f.flush()
                print(f"[INFO] {function_count}/{total_functions} - {func_name} processed "
                      f"in {time.time() - func_start_time:.2f}s")

                if function_count % 50 == 0:
                    ida_hexrays.clear_cached_cfuncs()
                    print(f"[INFO] Cleared decompiler cache at function {function_count}")

        ida_hexrays.clear_cached_cfuncs()
        print(f"[SUCCESS] Finished {function_count} functions, skipped {skipped_functions} "
              f"in {time.time() - start_time:.2f}s. Output saved to {self.output_file}")


# ========== Action Handler ==========
class StartDecompileHandler(idaapi.action_handler_t):
    def __init__(self, server_instance):
        idaapi.action_handler_t.__init__(self)
        self.server = server_instance

    def activate(self, ctx):
        self.server.start()
        return 1

    def update(self, ctx):
        return idaapi.AST_ENABLE_ALWAYS


# ========== Frida Generator Action Handlers ==========
class GenerateFridaCurrentHandler(idaapi.action_handler_t):
    def __init__(self, generator_instance):
        idaapi.action_handler_t.__init__(self)
        self.generator = generator_instance

    def activate(self, ctx):
        self.generator.generate_for_current_function()
        return 1

    def update(self, ctx):
        return idaapi.AST_ENABLE_ALWAYS



# ========== 菜单注册 ==========
server = AsyncDecompileServer()  # 全局变量, 原因：_worker 被立即调度执行、涉及文件和 Hex-Rays，临时对象可能在调度过程中被 GC 或线程调度机制打断
frida_generator = FridaCodeGenerator()  # 全局变量，同样原因


def register_decompile_menu():
    # server = AsyncDecompileServer()
    idaapi.create_menu("Edit/MCP", "MCP")

    idaapi.register_action(idaapi.action_desc_t(
        "decompile:start_async",
        "Start Async Decompile",
        StartDecompileHandler(server),
        "Ctrl-Alt-D",
        "Decompile all functions asynchronously"
    ))
    idaapi.attach_action_to_menu("Edit/MCP/Start_Decompile", "decompile:start_async", idaapi.SETMENU_APP)


def register_frida_menu():
    """注册Frida代码生成菜单"""
    idaapi.create_menu("Edit/MCP", "MCP")
    
    # 为当前函数生成Frida Hook
    idaapi.register_action(idaapi.action_desc_t(
        "frida:generate_current",
        "Generate Frida Hook (Current Function)",
        GenerateFridaCurrentHandler(frida_generator),
        "Ctrl-Alt-F",
        "Generate Frida hook code for current function"
    ))
    idaapi.attach_action_to_menu("Edit/MCP/Generate_Frida_Current", "frida:generate_current", idaapi.SETMENU_APP)


# 脚本加载时直接注册菜单
register_decompile_menu()
register_frida_menu()
