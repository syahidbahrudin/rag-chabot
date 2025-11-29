import Chat from "../components/Chat";
import Link from "next/link";

export default function Page() {
  return (
    <div className="h-screen w-screen flex flex-col  overflow-hidden">
      <Link
        className="absolute top-3 right-3 text-white bg-neutral-800 px-4 py-2 rounded-md"
        href="/admin"
      >
        Admin Panel
      </Link>
      <Chat />
    </div>
  );
}
