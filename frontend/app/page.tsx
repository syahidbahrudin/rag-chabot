import Chat from "../components/Chat";
import AdminPanel from "../components/AdminPanel";

export default function Page() {
  return (
    <div className="h-screen w-screen flex flex-col bg-neutral-950 overflow-hidden">
      <AdminPanel />
      <Chat />
    </div>
  );
}
