"use client";

import { useRouter } from "next/navigation";
import { CheckoutPage } from "@/components/CheckoutPage";

export default function CheckoutRoute() {
  const router = useRouter();
  return (
    <CheckoutPage
      onBack={() => router.push("/")}
      onSuccess={(orderId) => router.push(`/success?order=${encodeURIComponent(orderId)}`)}
    />
  );
}
